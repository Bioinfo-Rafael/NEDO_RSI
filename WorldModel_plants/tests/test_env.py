"""The online API must preserve the original substep physics and diagnostics."""
import jax.numpy as jnp
import numpy as np

from wmf import reward as rewardmod
from wmf.actions import clip_action
from wmf.plants.encode import encode_action
from wmf.sim.env import IncineratorEnv
from wmf.sim.rollout import coupled_step


def test_compiled_online_step_matches_original_loop(cfg):
    env = IncineratorEnv(cfg, seed=4, save_every=3)
    carry = (env.q, env.bed)
    for _ in range(2):
        action = clip_action(cfg, env.sample_action())
        field = jnp.asarray(encode_action(env.geom, action))
        unburnt_sum = 0.0
        for _ in range(env.save_every):
            carry, (_, ub) = coupled_step(carry, field, env.static, env.prm)
            unburnt_sum = unburnt_sum + ub
        expected = rewardmod.proxies(carry[0], unburnt_sum / env.save_every, env.static)
        energy = (field[..., 0].sum() * env.static["dx"]
                  + field[..., 1].sum() * env.static["dy"])
        obs, reward, info = env.step(action)
        np.testing.assert_allclose(obs["q"], np.asarray(carry[0]), rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(obs["bed"], np.asarray(carry[1]), rtol=2e-5, atol=2e-5)
        for name, value in expected.items():
            np.testing.assert_allclose(info[name], float(value), rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(reward, float(rewardmod.reward(expected, energy)),
                                   rtol=2e-5, atol=2e-5)
    assert env.t == 2
