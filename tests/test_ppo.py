"""Numerical checks for PPO objectives and the robot interface."""

import unittest
from pathlib import Path

import numpy as np
import torch

from exercises.ex3_ppo import PPOAgent
from rl.buffers import RolloutBatch


class PPOChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.agent = PPOAgent(
            19, 6, [32, 32], n_steps=8, mini_batch_size=4, n_epochs=2,
            clip_ratio=0.2, entropy_coeff=0.01,
        )

    def test_kl_matches_torch_distribution_reference(self):
        old_mu = torch.randn(8, 6)
        new_mu = old_mu + 0.15
        old_std = torch.rand(8, 6) + 0.3
        new_std = old_std * 1.2
        reference = torch.distributions.kl_divergence(
            torch.distributions.Normal(old_mu, old_std),
            torch.distributions.Normal(new_mu, new_std),
        ).sum(-1).mean()
        actual = self.agent.compute_kl_mean(old_mu, old_std, new_mu, new_std)
        torch.testing.assert_close(actual, reference)
        identical = self.agent.compute_kl_mean(old_mu, old_std, old_mu, old_std)
        torch.testing.assert_close(identical, torch.tensor(0.0))

    def test_surrogate_clips_only_the_favorable_direction(self):
        # Positive advantage / high ratio, negative / low ratio: saturated.
        # Positive / low ratio, negative / high ratio: still penalized.
        logp = torch.tensor([1.5, 0.5, 0.5, 1.5]).log().requires_grad_()
        adv = torch.tensor([2.0, -2.0, 2.0, -2.0])
        loss = self.agent.compute_surrogate_loss(logp, torch.zeros(4), adv)
        torch.testing.assert_close(loss, torch.tensor(0.3))
        loss.backward()
        torch.testing.assert_close(logp.grad[:2], torch.zeros(2))
        self.assertLess(logp.grad[2].item(), 0)
        self.assertGreater(logp.grad[3].item(), 0)

    def test_value_clipping_and_entropy(self):
        loss = self.agent.compute_value_loss(
            torch.tensor([5.0]), torch.tensor([4.0]), torch.tensor([6.0]),
        )
        torch.testing.assert_close(loss, torch.tensor(3.24))
        entropy = self.agent.compute_entropy_loss(torch.tensor([1.0, 3.0]))
        torch.testing.assert_close(entropy, torch.tensor(-0.02))

    def test_action_sampling_and_update(self):
        obs = torch.randn(8, 19)
        outputs = self.agent.select_action(obs)
        action, clipped, value, logp, mu, std = [x.clone() for x in outputs]
        self.assertEqual(action.shape, (8, 6))
        self.assertEqual(value.shape, (8,))
        self.assertEqual(logp.shape, (8,))
        torch.testing.assert_close(clipped, action.clamp(-1, 1))
        reference = torch.distributions.Normal(mu, std).log_prob(action).sum(-1)
        torch.testing.assert_close(logp, reference)
        adv = torch.linspace(-1, 1, 8)
        batch = RolloutBatch(
            obs=obs, act=action, logp=logp, mu=mu, std=std,
            val=value, ret=value + adv, adv=adv,
        )
        actor_before = [p.detach().clone() for p in self.agent.actor.parameters()]
        critic_before = [p.detach().clone() for p in self.agent.critic.parameters()]
        stats = self.agent.update(batch)
        for result in vars(stats).values():
            self.assertTrue(torch.isfinite(torch.as_tensor(result)).all())
        for module, before in [(self.agent.actor, actor_before), (self.agent.critic, critic_before)]:
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, module.parameters())))
            for parameter in module.parameters():
                self.assertTrue(torch.isfinite(parameter).all())
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_robot_reset_and_step(self):
        from envs.so100_rl_env import SO100RLEnv
        from rl.common import set_seed

        set_seed(42)
        root = Path(__file__).resolve().parents[1]
        env = SO100RLEnv(root / "assets/mujoco/so100_pos_ctrl.xml")
        try:
            obs, _ = env.reset()
            next_obs, reward, terminated, truncated, info = env.step(np.zeros(6, np.float32))
            self.assertEqual(obs.shape, (19,))
            self.assertEqual(next_obs.shape, (19,))
            self.assertTrue(np.isfinite(next_obs).all())
            self.assertTrue(np.isfinite(reward))
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertIn("ee_tracking_error", info)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
