"""2D reacting-gas solver — the ground truth of this project.

Scope note
----------
This simulator *defines* truth for the demo.  It is deliberately NOT a faithful
model of a real incinerator: the goal is a self-consistent, deterministic,
cheap world whose dynamics are rich enough that a learned world model has
something non-trivial to capture, and whose control regimes an agent can
*rediscover*.  Fidelity beyond that is out of scope.

Regimes deliberately built in (what the agent should be able to find):
  * stoker too fast          -> waste leaves the grate unburnt
  * primary air too low      -> O2 starvation, unburnt fuel, low temperature
  * primary air too high     -> convective quenching, low temperature
  * secondary air placement  -> burns out volatiles in the freeboard
  * a sweet spot in between  -> hot, clean, low unburnt

Numerics
--------
Artificial compressibility (``dp/dt = -c^2 div(u)``) so the whole step is
explicit and ``jit``/``scan``/``vmap`` go through cleanly - no pressure Poisson
solve.  2nd-order upwind advection, centred diffusion, forward Euler with the
reaction term sub-cycled (the Arrhenius source is the stiff part).

Grid convention: ``[j, i]``, ``j = 0`` at the bottom.  See API.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------- constants
# state channel order (API.md 3)
T, U, V, P, YF, YO2, YP = 0, 1, 2, 3, 4, 5, 6
N_Q = 7

# Boussinesq reference.  This is the *operating* temperature of the furnace,
# NOT ambient: the expansion is linearised about the state the gas actually
# sits at.  Using 300 K here inflates the buoyancy force by ~3.7x and drives
# the flow to 40-50 m/s, which is both non-physical (a real furnace runs at
# 1-10 m/s) and breaks the artificial-compressibility assumption c >> |u|.
T_REF = 1100.0       # K   Boussinesq reference (furnace operating point)
T_AMB = 300.0        # K   ambient, used only for reporting
T_AIR = 320.0        # K   injected air temperature
P_REF = 1.0          # kg/m3-ish reference density scale (Boussinesq)
RHO0 = 0.35          # kg/m3  hot flue gas (Boussinesq reference)
RHO_AIR = 1.29       # kg/m3  injected air at normal conditions
CP = 1150.0          # J/kgK
# Eddy viscosity, not molecular: the furnace is turbulent and we do not resolve
# it, so a mixing-length-scale value is both cheaper and more honest than 1.6e-4.
NU = 1.0e-1          # m2/s
PR = 0.7             # Prandtl
SC = 0.7             # Schmidt
GRAV = 9.81          # m/s2
# Artificial sound speed. Lowered from 40 so that the explicit step can be
# 3 ms instead of 1 ms: a realistic grate residence is ~700 s, which at 1 ms
# costs 10 minutes of wall clock per episode and makes the dataset unaffordable.
# c/|u| is 2.2 at the measured 9.2 m/s, which still separates the pseudo-acoustic
# modes from the flow. Going to c=12 buys another 60% but drops the ratio to 1.3,
# which is too close to call.
C_SOUND = 20.0       # m/s
# Rayleigh drag standing in for unresolved wall friction and internal blockage.
# Without it, buoyancy alone drives |u| past C_SOUND and the artificial-
# compressibility assumption breaks.  v_terminal ~ g*dT/T0 / C_DRAG.
C_DRAG = 2.0         # 1/s

# Radiation multiplies the configured convective wall coefficient; see
# SimParams.from_plant for the arithmetic.
WALL_RAD_FACTOR = 6.0
# Outlet pressure relaxation.  Hard-setting p = 0 at the flue puts the full
# interior pressure across one cell, which accelerates the gas past C_SOUND in
# a single step; relaxing instead keeps the outflow well behaved.
OUTLET_RELAX = 0.08
# Fraction of the injected volume that the flue is driven to carry.  Relaxing
# the pressure alone made the flue a weak sink - only ~15% of the injected air
# left through it - and with no real suction the *position* of the flue stopped
# affecting the flow at all, which would leave a geometry-conditioned model
# with nothing to condition on.  Setting the outflow velocity from the global
# volume balance makes the flue the sink it should be.
OUTLET_GAIN = 1.0
# Artificial-compressibility damping.  Nothing else damps the pseudo-acoustic
# modes: the volume sources pump them, the pressure amplitude reaches ~1.5 kPa
# and, divided by the small Boussinesq reference density, that gradient
# accelerates the gas by ~2e4 m/s^2 - i.e. the velocity field becomes acoustic
# noise rather than flow.  A bulk (pressure) diffusion term damps them without
# changing the balance the mean field settles into.
# Expressed as a diffusion CFL number rather than a fixed m2/s so it scales
# with the grid: the stability limit is 4*P_DIFF*dt/min(dx,dy)^2 < 1, and the
# plants differ in dy by more than 2x.  0.1 was measured to remove the acoustic
# noise entirely (|u| 37 -> 8.9 m/s, |p| 1467 -> 56 Pa) with ~2x margin; 0.5
# diverges.
# Pressure diffusion, as a fraction of the explicit 2D stability limit
#     nu_p * dt * (1/dx^2 + 1/dy^2) <= 1/2.
# It used to be written as 0.30 * min(dx,dy)^2 / dt, which is 0.40 of the limit
# on plant_a's 0.19 x 0.11 m cells and 0.60 - unstable - on plant_c's square
# 0.14 m cells; plant_c's pressure went NaN at 0.8 s while a and b ran. Writing
# it against the actual criterion makes it aspect-ratio independent and leaves
# plant_a's damping unchanged.
P_DIFF_FRAC = 0.80
# Emergency guard only.  If this ever binds the physics is wrong, not the flow;
# tests/test_stability.py asserts that it does not bind.
U_CLAMP = 0.8 * C_SOUND

Y_O2_AIR = 0.232     # mass fraction of O2 in air
S_STOICH = 3.0       # kg O2 per kg volatile
A_ARR = 1.0e5        # 1/s  pre-exponential
EA_R = 8000.0        # K    activation temperature (smoothed: real is ~2x this,
                     #      which would make the source term far too stiff)
N_REACT_SUB = 8      # reaction sub-steps per gas step

ALPHA = NU / PR      # thermal diffusivity
DIFF = NU / SC       # species diffusivity


@dataclass(frozen=True)
class SimParams:
    dt: float = 3.0e-3
    n_sub: int = N_REACT_SUB
    q_cp: float = 9.0e6 / CP     # K per unit mass fraction burnt (from waste LHV)
    lam_loss: float = 0.01       # 1/s  wall heat loss rate
    t_wall: float = 600.0        # K

    @classmethod
    def from_plant(cls, cfg, **overrides) -> "SimParams":
        """Derive the solver parameters that differ between plants.

        Without this the plants differ only in geometry and waste composition,
        so wall.heat_transfer / wall.temperature / waste.lhv - exactly the
        quantities a plant-level latent is supposed to identify - are constant
        across the whole dataset and cannot be identified even in principle.
        """
        # h [W/m2K] over a characteristic gas column -> a bulk loss rate [1/s].
        # The configured coefficient is convective only; in a furnace the wall
        # takes several times more by radiation. At 1100 K gas against a 600 K
        # waterwall the radiative coefficient is
        #   eps*sigma*(T^4 - Tw^4)/(T - Tw) = 121 W/m2K
        # against 25 convective, so the effective coefficient is ~6x. This is
        # the term that carries the heat to the steam side, and it has to be
        # right in magnitude because the alternative - letting the diffusion
        # stencil see a fixed wall temperature - is off by a factor of four.
        lam = (WALL_RAD_FACTOR * cfg.wall.heat_transfer
               / (RHO0 * CP * cfg.furnace.height))
        kw = dict(
            q_cp=cfg.waste.lhv * 1.0e6 / CP,
            lam_loss=float(lam),
            t_wall=float(cfg.wall.temperature),
        )
        kw.update(overrides)
        return cls(**kw)


# ---------------------------------------------------------------- operators
def _d_dx(f, dx):
    return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dx)


def _d_dy(f, dy):
    return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dy)


def _lap(f, dx, dy):
    return (
        (jnp.roll(f, -1, axis=1) - 2.0 * f + jnp.roll(f, 1, axis=1)) / dx**2
        + (jnp.roll(f, -1, axis=0) - 2.0 * f + jnp.roll(f, 1, axis=0)) / dy**2
    )


def _upwind(f, u, v, dx, dy):
    """2nd-order-ish upwind advection: -(u . grad) f."""
    fxm = (f - jnp.roll(f, 1, axis=1)) / dx
    fxp = (jnp.roll(f, -1, axis=1) - f) / dx
    fym = (f - jnp.roll(f, 1, axis=0)) / dy
    fyp = (jnp.roll(f, -1, axis=0) - f) / dy
    return -(jnp.where(u > 0, u * fxm, u * fxp) + jnp.where(v > 0, v * fym, v * fyp))


def reaction_rate(t, yf, yo2):
    """Single-step Arrhenius: F + s O2 -> P + Q."""
    return A_ARR * jnp.clip(yf, 0.0) * jnp.clip(yo2, 0.0) * jnp.exp(-EA_R / jnp.maximum(t, 1.0))


def react(t, yf, yo2, yp, dt, prm: SimParams):
    """Sub-cycled single-step combustion.

    Split out of ``gas_step`` so the stoichiometry and the energy release can be
    asserted on directly (tests/test_conservation.py) rather than inferred from
    a full coupled run.  Returns ``((T, Y_F, Y_O2, Y_P), mean_rate)``.
    """
    sub = dt / prm.n_sub

    def _step(carry, _):
        t_, yf_, yo_, yp_ = carry
        w = reaction_rate(t_, yf_, yo_)
        # never consume more than is present within one sub-step
        w = jnp.maximum(jnp.minimum(w, jnp.minimum(yf_, yo_ / S_STOICH) / sub), 0.0)
        return (t_ + sub * prm.q_cp * w,
                yf_ - sub * w,
                yo_ - sub * S_STOICH * w,
                yp_ + sub * (1.0 + S_STOICH) * w), w

    carry, w_hist = jax.lax.scan(_step, (t, yf, yo2, yp), None, length=prm.n_sub)
    return carry, w_hist.mean(axis=0)


# ---------------------------------------------------------------- one step
def gas_step(q, a, fuel_src, heat_src, static, prm: SimParams, t_air_in=None):
    """Advance the gas state one ``prm.dt``.

    ``q``      [H, W, 7]  state
    ``a``      [H, W, 4]  actuator flux-density field (see plants.encode)
    ``fuel_src`` [H, W] volatile release rate from the bed [1/s]
    ``heat_src`` [H, W] char-combustion heat from the bed [K/s].  It is applied
        here, before the clip, so that no source can push the state past the
        bounds - adding it downstream silently defeats the clip.
    ``t_air_in`` [H, W] temperature of the injected air. Primary air has passed
        through the bed and arrives at the bed temperature; secondary air is
        ambient. Defaults to ambient everywhere.
    ``static`` dict of precomputed static arrays (mask, injection maps, dx, dy)
    """
    mask = static["mask"]
    dx, dy = static["dx"], static["dy"]

    t, u, v, p = q[..., T], q[..., U], q[..., V], q[..., P]
    yf, yo2, yp = q[..., YF], q[..., YO2], q[..., YP]

    # --- momentum -----------------------------------------------------
    buoy = GRAV * (t - T_REF) / T_REF
    du = _upwind(u, u, v, dx, dy) - _d_dx(p, dx) / RHO0 + NU * _lap(u, dx, dy) - C_DRAG * u
    dv = _upwind(v, u, v, dx, dy) - _d_dy(p, dy) / RHO0 + NU * _lap(v, dx, dy) + buoy - C_DRAG * v


    # --- transport of scalars (reaction handled separately) -----------
    dt_ = (_upwind(t, u, v, dx, dy) + ALPHA * _lap(t, dx, dy)
           - prm.lam_loss * static["loss_scale"] * (t - static["t_wall"])
           + heat_src)
    dyf = _upwind(yf, u, v, dx, dy) + DIFF * _lap(yf, dx, dy)
    dyo = _upwind(yo2, u, v, dx, dy) + DIFF * _lap(yo2, dx, dy)
    dyp = _upwind(yp, u, v, dx, dy) + DIFF * _lap(yp, dx, dy)

    # --- actuator sources ---------------------------------------------
    # `a` holds a volumetric flux density [m/s]; dividing by the cell height or
    # width turns it into a volumetric dilution rate [1/s].  The density ratio
    # accounts for cold dense air displacing hot light gas - without it the
    # Boussinesq single density under-delivers O2 by a factor of ~3.7.
    rho_ratio = RHO_AIR / RHO0
    f_pri = a[..., 0] / dy * rho_ratio     # primary air, upward through the grate
    f_sec = a[..., 1] / dx * rho_ratio     # secondary air, horizontal
    f_fuel = fuel_src               # volatile release from the bed [1/s]

    t_in = T_AIR if t_air_in is None else t_air_in
    dyo = dyo + (Y_O2_AIR - yo2) * (f_pri + f_sec)
    dt_ = dt_ + (t_in - t) * f_pri + (T_AIR - t) * f_sec
    dyf = dyf + (1.0 - yf) * f_fuel
    dyp = dyp - yp * (f_pri + f_sec)
    dv = dv + (a[..., 0] - v) * f_pri
    du = du + (static["sec_dir"] * a[..., 1] - u) * f_sec

    # --- artificial compressibility -----------------------------------
    # The injected air is a genuine volume source, so it belongs in the
    # divergence constraint; that is what makes the air actually drive flow.
    div = _d_dx(u, dx) + _d_dy(v, dy)
    p_diff = P_DIFF_FRAC * 0.5 / (prm.dt * (1.0 / dx**2 + 1.0 / dy**2))
    dp = -(C_SOUND**2) * RHO0 * (div - (f_pri + f_sec)) + p_diff * _lap(p, dx, dy)

    # --- integrate transport ------------------------------------------
    dt = prm.dt
    u = u + dt * du
    v = v + dt * dv
    p = p + dt * dp
    t = t + dt * dt_
    yf = yf + dt * dyf
    yo2 = yo2 + dt * dyo
    yp = yp + dt * dyp

    # --- reaction, sub-cycled (the stiff part) -------------------------
    (t, yf, yo2, yp), omega = react(t, yf, yo2, yp, dt, prm)

    # --- boundaries ----------------------------------------------------
    u = u * mask
    v = v * mask
    # Solid cells are extended with zero gradient, not pinned to the wall
    # temperature. Pinning them made every wall-adjacent fluid cell exchange
    # with a fixed bath at the turbulent diffusivity: ALPHA/dy^2 = 11.4 1/s
    # across roughly 200 perimeter cells, which drains 9.4 MW against a 3.5 MW
    # fuel input. Real wall heat transfer for this furnace is ~2.5 MW
    # (0.45 convective + 2.0 radiative), and that is what ``lam_loss`` is for.
    nbr = (jnp.roll(t * mask, 1, 0) + jnp.roll(t * mask, -1, 0)
           + jnp.roll(t * mask, 1, 1) + jnp.roll(t * mask, -1, 1))
    wgt = (jnp.roll(mask, 1, 0) + jnp.roll(mask, -1, 0)
           + jnp.roll(mask, 1, 1) + jnp.roll(mask, -1, 1))
    t_solid = jnp.where(wgt > 0, nbr / jnp.maximum(wgt, 1e-6), static["t_wall"])
    t = jnp.where(mask > 0, t, t_solid)
    yf = jnp.where(mask > 0, jnp.clip(yf, 0.0, 1.0), 0.0)
    yo2 = jnp.where(mask > 0, jnp.clip(yo2, 0.0, Y_O2_AIR), 0.0)
    yp = jnp.where(mask > 0, jnp.clip(yp, 0.0, 1.0), 0.0)
    p = jnp.where(mask > 0, p, 0.0)
    p = p - static["outlet"] * OUTLET_RELAX * p          # flue relaxes to ambient

    # Convective outflow: what is injected has to leave, and it leaves here.
    outlet = static["outlet"]
    q_inj = ((f_pri + f_sec) * dx * dy).sum()            # m3/s per unit depth
    v_out = OUTLET_GAIN * q_inj / jnp.maximum(outlet.sum() * dx, 1e-9)
    v = jnp.where(outlet > 0, v_out, v)
    u = jnp.clip(u, -U_CLAMP, U_CLAMP)
    v = jnp.clip(v, -U_CLAMP, U_CLAMP)
    t = jnp.clip(t, 250.0, 2500.0)

    q_new = jnp.stack([t, u, v, p, yf, yo2, yp], axis=-1)
    return q_new, omega


def initial_state(static, t_init: float = 1100.0):
    """Quiescent furnace, pre-heated so that the first episode can ignite."""
    h, w = static["mask"].shape
    mask = static["mask"]
    q = jnp.zeros((h, w, N_Q))
    q = q.at[..., T].set(jnp.where(mask > 0, t_init, 600.0))
    q = q.at[..., YO2].set(jnp.where(mask > 0, Y_O2_AIR, 0.0))
    return q
