import tensorflow as tf
import numpy as np

import tempfile
import unittest
from pathlib import Path

from train import RobotAgent, configure, gae, make_env


class CorrectnessTests(unittest.TestCase):
    def test_termination_truncation_and_vector_boundaries(self):
        values = np.array([[2, 10], [3, 20], [4, 30]], np.float32)
        next_values = np.array([[3, 11], [99, 21], [5, 31]], np.float32)
        rewards = np.ones((3, 2), np.float32)
        terminated = np.array([[0, 0], [1, 0], [0, 0]], np.float32)
        truncated = np.array([[0, 1], [0, 0], [0, 1]], np.float32)
        adv, _ = gae(rewards, values, next_values, terminated, truncated, .9, .8)
        self.assertAlmostEqual(float(adv[1, 0]), -2.)  # Fall: no value from next state.
        self.assertAlmostEqual(float(adv[0, 1]), .9, places=5)  # Timeout bootstraps, stops recursion.
        self.assertAlmostEqual(float(adv[2, 1]), -1.1, places=5)
        self.assertAlmostEqual(float(adv[0, 0]), 1+.9*3-2+.9*.8*(-2), places=5)
        self.assertAlmostEqual(float(adv[2, 0]), 1.5, places=5)  # Rollout boundary bootstraps.

    def test_environment_contract(self):
        env = make_env()
        try:
            obs, _ = env.reset(seed=123)
            self.assertEqual(obs.shape, (28,))
            for step in range(50):
                obs, reward, terminated, truncated, info = env.step(np.zeros(3, np.float32))
                self.assertAlmostEqual(reward, -np.linalg.norm(obs[3:6]-obs[25:]), places=5)
                self.assertFalse(terminated)
                self.assertEqual(truncated, step == 49)
            self.assertIn('is_success', info)
        finally:
            env.close()

    def test_shaping_preserves_physics_and_native_evaluation(self):
        native, shaped = make_env(), make_env(approach_weight=.5)
        rng = np.random.default_rng(13)
        try:
            a, _ = native.reset(seed=654)
            b, _ = shaped.reset(seed=654)
            np.testing.assert_array_equal(a, b)
            for _ in range(50):
                action = rng.uniform(-1, 1, 3).astype(np.float32)
                a, ra, ta, xa, ia = native.step(action)
                b, rb, tb, xb, ib = shaped.step(action)
                np.testing.assert_array_equal(a, b)
                self.assertEqual((ta, xa), (tb, xb))
                self.assertAlmostEqual(ra, -np.linalg.norm(a[3:6]-a[25:]), places=5)
                self.assertAlmostEqual(rb, ra-.5*np.linalg.norm(b[:3]-b[3:6]), places=6)
                self.assertAlmostEqual(ib['native_reward'], ra, places=7)
                self.assertEqual(ia['is_success'], ib['is_success'])
        finally:
            native.close()
            shaped.close()

    def test_distribution_update_and_checkpoint(self):
        agent = RobotAgent()
        obs = np.random.default_rng(7).normal(size=(64, 28)).astype(np.float32)
        action, latent, lp, values = agent.predict(obs, stochastic=True)
        out = agent._actor(obs).numpy()
        expected = agent._gaussian_log_prob(latent, out[:, :3], np.clip(out[:, 3:], -5, 2)).numpy()
        np.testing.assert_allclose(lp, expected, atol=3e-5)
        np.testing.assert_allclose(agent.dense(obs, agent.actor_arrays), out, atol=2e-6)
        # Fixed positive advantage must increase likelihood of these actions.
        before = lp.mean()
        metrics = agent.train_rollout(*[tf.constant(x, tf.float32) for x in
            (obs, latent, lp, np.ones(64), values)]).numpy()
        agent.sync_numpy()
        out = agent._actor(obs).numpy()
        after = agent._gaussian_log_prob(latent, out[:, :3], np.clip(out[:, 3:], -5, 2)).numpy().mean()
        self.assertGreater(after, before)
        self.assertTrue(np.all(np.isfinite(metrics)))
        # Retained latent keeps saturated action log probabilities finite.
        saturated = np.full((64, 3), 15., np.float32)
        self.assertTrue(np.isfinite(agent._gaussian_log_prob(saturated, out[:, :3], np.clip(out[:, 3:], -5, 2))).all())
        agent.obs_stats.update(obs)
        expected_action = agent.predict(agent.obs_stats.normalize(obs))[0]
        with tempfile.TemporaryDirectory() as tmp:
            agent.save_weights(tmp)
            other = RobotAgent()
            other.load_weights(Path(tmp))
            np.testing.assert_array_equal(expected_action, other.predict(other.obs_stats.normalize(obs))[0])
            for a, b in zip(agent._optimizer.variables, other._optimizer.variables):
                np.testing.assert_array_equal(a.numpy(), b.numpy())
        self.assertEqual(len(agent._cache_obs), 0)  # Inference never contaminates training cache.


if __name__ == '__main__':
    configure()
    unittest.main()
