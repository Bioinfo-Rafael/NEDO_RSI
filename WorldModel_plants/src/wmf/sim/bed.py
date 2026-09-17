"""1D stoker bed model, coupled to the 2D gas as a bottom boundary source.

Without this, primary-air zone control has no meaning and the thing is not a
stoker furnace.  It is 1D along the grate, so it costs essentially nothing.

    d/dt [m_w, m_v, m_c] + v_stoker d/dx [m_w, m_v, m_c] = [-R_dry, ..., ...]

    drying     : m_w -> steam            (gated on surface gas temperature)
    pyrolysis  : m_v -> volatiles to gas (gated on temperature)
    char burn  : m_c -> heat + CO2       (gated on temperature AND local O2)

The volatile flux becomes the fuel source of the gas phase; char combustion
releases heat directly into the first gas cell above the grate.

Bed channel order: 0 = m_w (moisture), 1 = m_v (volatile), 2 = m_c (char).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .physics import Y_O2_AIR

# No compression. This was 4.0, which together with an unrealistically fast
# grate put our residence at 16.7 s against the 11-12 minutes measured on the
# real plant's August log - a factor of 42. Compressing the slow manifold made
# episodes cheap but moved the one time scale that matters most for control
# away from the plant we are trying to represent.
BED_TIME_SCALE = 1.0

# Rate constants [1/s]. Calibrated against the two numbers the real plant's log
# pins down: a grate residence of 660-720 s and unburnt carbon in the ash below
# 3%. For first-order kinetics those together require K * tau_res ~ 3.5, i.e.
# K ~ 0.006 at the nominal 600 s residence, with drying running ahead of
# pyrolysis and pyrolysis ahead of char burnout so the three zones stay staged
# along the grate. The values that came out of the old compressed time scale
# (0.0015-0.0025) left 58% of the feed leaving the grate unburnt.
#
# The inventory these imply (~30 kg/m2) is the model's inventory, not a claim
# about physical bed depth: a real 0.3-0.8 m bed is deep because MSW is bulky,
# and a lumped first-order bed cannot hold that mass and still convert it.
K_DRY = 0.012     # tau  83 s
K_PYRO = 0.008    # tau 125 s
K_CHAR = 0.006    # tau 167 s

# Onset temperatures, at their measured values rather than the conservative
# ones this started with (450 / 700 / 850). All three were high enough to keep
# the bed's own cold branch absorbing: at lambda 1.73 the bed settled at 714 K,
# below a char ignition set at 850, so the primary air arrived barely preheated
# and the freeboard could not ignite either.
T_DRY = 380.0     # K  drying, just above the boiling point of water
T_PYRO = 550.0    # K  MSW devolatilisation runs over roughly 500-700 K
T_CHAR = 750.0    # K  char ignition in a packed bed, 650-750 K
T_WIDTH = 60.0    # K  smoothing width of each onset (keeps the source non-stiff)

Q_CHAR = 3.3e7    # J/kg  char heat of combustion
CP_GAS = 1150.0
S_CHAR = 2.67     # kg O2 per kg char (C + O2 -> CO2)

# --- bed energy -------------------------------------------------------------
# The bed needs its own temperature. Driving the drying and pyrolysis gates off
# the gas temperature at the grate surface is wrong in a way that only shows up
# once the residence is realistic: the primary air enters the freeboard exactly
# there, so that cell sits near the air temperature, the gates close, and the
# furnace goes out no matter how hot it started. Measured: T_surf fell from
# 1727 K to 647 K in 0.6 s while the bulk gas was still at 1632 K.
#
# Two things follow from giving the bed a temperature, and both are physical:
# the solids have thermal inertia on the order of the residence, and the primary
# air is preheated by the bed on its way through rather than arriving cold.
CP_SOLID = 1200.0    # J/kgK
SIGMA = 5.67e-8      # Stefan-Boltzmann
EMIS = 0.8           # bed emissivity
L_VAP = 2.26e6       # J/kg  latent heat of vaporisation
H_PYRO = 1.0e6       # J/kg  endothermic heat of pyrolysis
T_B_INIT = 400.0     # K

# Primary air arrives from a steam air preheater, not from the yard. Every
# municipal incinerator has one, for exactly the reason it shows up here: it is
# what closes the bed's energy balance and what dries wet waste. Omitting it
# left the bed paying 1368 kW/m2 to heat its own air from 320 K against
# 1420 kW/m2 of char combustion and 262 kW/m2 of evaporation - a 210 kW/m2
# deficit, so the grate cooled at 1.3 K/s from any starting temperature.
# 450 K = 177 C, mid-range for the 150-250 C these preheaters run at.
T_AIR_PREHEAT = 450.0   # K

# Flame spread along the bed. The ignition front on a grate travels back
# against the grate motion at roughly 1-10 mm/s, carried by conduction and by
# radiation between neighbouring bed cells. Without it nothing propagates
# upstream and the feed end can only be lit from above. As a diffusivity,
# D ~ v_front * front thickness ~ 0.005 m/s * 0.3 m.
D_BED = 1.5e-3       # m2/s


def _gate(t, t_on, width=T_WIDTH):
    """Smooth ignition switch. A hard threshold makes the coupled system stiff
    and the trajectories non-smooth, which hurts both the solver and the
    learnability of the dynamics."""
    return jax.nn.sigmoid((t - t_on) / width)


def bed_step(bed, t_gas4, o2_surf, o2_supply, air_mass, stoker_speed, waste_feed,
             static, dt):
    """Advance the bed one ``dt``.

    ``bed``      [W, 4]  moisture / volatile / char [kg/m2] and bed temperature [K]
    ``t_gas4``   [W]     column mean of T^4 in the freeboard [K^4], for radiation.
                         T^4 averaged, not T: see the note in ``coupled_step``.
    ``o2_surf``  [W]     gas O2 mass fraction there
    ``o2_supply``[W]     O2 delivered by primary air in that column [kg/m2/s]
    ``air_mass`` [W]     primary air mass flux through that column [kg/m2/s]
    returns (bed_new, vol_flux, char_heat_to_gas, unburnt_out, t_air_out)
    """
    dx = static["dx"]
    on_grate = static["grate_cols"]          # [W] 1 where the stoker runs

    m_w, m_v, m_c, t_b = bed[:, 0], bed[:, 1], bed[:, 2], bed[:, 3]

    # --- rates, driven by the bed's own temperature --------------------
    r_dry = K_DRY * m_w * _gate(t_b, T_DRY)
    r_pyro = K_PYRO * m_v * _gate(t_b, T_PYRO)
    # Char burns inside the bed, with the O2 of the primary air arriving from
    # underneath at 23.2% -- not with whatever O2 the freeboard happens to have.
    # Scaling by ``o2_surf`` both double-counted the O2 limit imposed two lines
    # below and pointed the wrong way: when the flame dims, freeboard O2 should
    # rise, so the bed burn should not follow the flame down. Measured, it did:
    # freeboard O2 0.232 -> 0.068 cut the char burn to 0.3x, q_char 958 -> 290
    # kW/m2 against 437 kW/m2 of air preheat, and the bed cooled until nothing
    # was left to gasify. The only O2 limit that belongs here is the supply one.
    r_char = K_CHAR * m_c * _gate(t_b, T_CHAR)
    r_char = jnp.minimum(r_char, o2_supply / S_CHAR)

    # --- transport along the grate (1st-order upwind, v >= 0) ----------
    v_eff = stoker_speed * BED_TIME_SCALE

    def _adv(m):
        return -v_eff * (m - jnp.roll(m, 1)) / dx

    # --- fresh waste injected at the feed end --------------------------
    # NOT scaled by BED_TIME_SCALE.  Scaling the transport and the kinetics by
    # the same factor leaves the steady-state release rate equal to the feed
    # rate and leaves K * residence (hence the burnout fraction) unchanged; the
    # bed simply holds 1/S as much mass.  Scaling the feed as well would cut the
    # air/fuel ratio by S and over-fire the furnace.
    feed = static["feed_profile"] * waste_feed          # [W] kg/m2/s
    w_frac = static["w_frac"]
    v_frac = static["v_frac"]
    c_frac = static["c_frac"]

    m_w = m_w + dt * (_adv(m_w) - r_dry + feed * w_frac)
    m_v = m_v + dt * (_adv(m_v) - r_pyro + feed * v_frac)
    m_c = m_c + dt * (_adv(m_c) - r_char + feed * c_frac)

    m_w = jnp.clip(m_w, 0.0) * on_grate
    m_v = jnp.clip(m_v, 0.0) * on_grate
    m_c = jnp.clip(m_c, 0.0) * on_grate

    # --- bed energy balance ---------------------------------------------
    # heated by radiation from the flame and by char burning inside the bed,
    # cooled by the primary air passing through it and by drying and pyrolysis
    q_rad = EMIS * SIGMA * (jnp.clip(t_gas4, 0.0) - jnp.clip(t_b, 0.0) ** 4)
    q_char_bed = r_char * Q_CHAR
    q_air = air_mass * CP_GAS * (t_b - T_AIR_PREHEAT)
    q_evap = r_dry * L_VAP + r_pyro * H_PYRO
    heat_cap = jnp.maximum((m_w + m_v + m_c) * CP_SOLID, 1.0e3)
    spread = D_BED * (jnp.roll(t_b, 1) + jnp.roll(t_b, -1) - 2.0 * t_b) / (dx * dx)
    t_b = t_b + dt * ((q_rad + q_char_bed - q_air - q_evap) / heat_cap
                      - v_eff * (t_b - jnp.roll(t_b, 1)) / dx
                      + spread * on_grate)
    t_b = jnp.clip(t_b, 280.0, 2000.0)

    # --- what leaves the end of the grate = unburnt loss ----------------
    last = static["last_grate_col"]
    unburnt_out = v_eff * (m_v[last] + m_c[last])

    # Char burns inside the bed, so its heat goes to the bed, not straight to
    # the gas. What the gas receives is the preheated primary air leaving the
    # bed at t_b, which is handled by the coupling.
    # ``q_rad`` is returned so the coupling can put it into the gas. It is a
    # two-way exchange and the bed is normally the hotter side, so leaving it
    # out of the gas equation is not a small approximation - it is an energy
    # leak. Measured at nominal: the bed took 6.4 MW of char heat, returned
    # 2.5 MW as preheated primary air, and radiated the remaining 3.9 MW into
    # nothing. The flue gas was fully burnt out (Y_F = 0) and still arrived at
    # 707 K where the lumped enthalpy balance asks for 1502 K.
    char_heat = jnp.zeros_like(r_char)
    return (jnp.stack([m_w, m_v, m_c, t_b], axis=-1),
            r_pyro, char_heat, unburnt_out, t_b, q_rad)


def steady_fill(static, waste_feed: float):
    """Areal densities [kg/m2] the kinetics require to release fuel as fast as
    it is fed.

    Picking a round ``fill`` and splitting it by the as-fed fractions does not
    do this: at 60 kg/m2 the volatile inventory came out 23.4 kg/m2 against the
    54 kg/m2 that K_PYRO needs to match a 2.5 kg/s feed, so the furnace started
    at 43% of its own firing rate and cooled from there. The residence also
    carries mass off the end of the grate, so this is an estimate rather than
    the exact fixed point, but it is the right order and the right ratio, and
    the totals it gives (~107 kg/m2, i.e. a 0.3 m bed at ~350 kg/m3) are what a
    stoker actually holds.
    """
    length = float(static["grate_cols"].sum()) * float(static["dx"])
    feed = waste_feed / max(length, 1e-9)          # kg/m2/s
    return (feed * float(static["w_frac"]) / K_DRY,
            feed * float(static["v_frac"]) / K_PYRO,
            feed * float(static["c_frac"]) / K_CHAR)


def initial_bed(static, waste_feed: float = 1.25, t_b: float = 1150.0):
    """The state automatic combustion control inherits from the start-up
    burners: a grate loaded to the inventory its own feed rate implies, and
    already burning.

    ``t_b`` is the operating bed temperature of a stoker grate (1100-1300 K),
    not a warm-but-unlit 700 K. Measured bed budget at 700 K: char burn +73,
    radiation +55, air preheat -200, evaporation/pyrolysis -111 kW/m2, i.e.
    -2.7 K/s. Below char ignition the bed cannot pay for the air it has to
    heat and the furnace goes out. That is a real operating limit the
    controller has to respect, not an initial condition.
    """
    g = static["grate_cols"]
    w = g.shape[0]
    m_w, m_v, m_c = steady_fill(static, waste_feed)
    m = jnp.zeros((w, 4))
    m = m.at[:, 0].set(g * m_w)
    m = m.at[:, 1].set(g * m_v)
    m = m.at[:, 2].set(g * m_c)
    m = m.at[:, 3].set(jnp.where(g > 0, t_b, 300.0))
    return m
