"""CPU checks for the mathematical contracts of the released modules."""

import unittest
from unittest.mock import patch

import torch

from surgview_vla import ActionHead, What2ViewV0, Where2ViewV1, Where2ViewV11


class CoreContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(19)

    def test_visibility_gated_pooling_of_constant_features(self):
        model = What2ViewV0(hidden_dim=32).eval()
        # Every visual position has the same feature; weighted pooling must
        # preserve it before the predicted visibility gate is applied.
        feature = torch.randn(2, 1024)
        grid = feature[:, None, None, :].expand(-1, 14, 24, -1)
        with torch.no_grad():
            result = model(torch.randn(2, 4096), grid)
        torch.testing.assert_close(result["z_op"], feature * result["visibility"][:, None])
        self.assertEqual(result["attention"].shape, (2, 45, 80))
        self.assertTrue(((result["attention"] >= 0) & (result["attention"] <= 1)).all())

    def test_view_difference_after_shared_denormalization(self):
        model = Where2ViewV1(
            hidden_dim=32, view_mean=(0.2, 0.3, -1.0, 0.4),
            view_scale=(2.0, 3.0, 4.0, 5.0),
        ).eval()
        with torch.no_grad():
            result = model(torch.randn(2, 4096), torch.randn(2, 1024), torch.rand(2, 45, 80))
        expected = (result["g_des_normalized"] - result["g_cur_normalized"]) * model.view_scale
        torch.testing.assert_close(result["e_view"], expected)
        torch.testing.assert_close(result["g_cur"], result["g_cur_normalized"] * model.view_scale + model.view_mean)

    def test_optional_scale_residual_preserves_frozen_base(self):
        base = Where2ViewV1(hidden_dim=32).eval()
        model = Where2ViewV11(base, hidden_dim=32)
        model.train()
        self.assertFalse(base.training)
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))
        inputs = (torch.randn(2, 4096), torch.randn(2, 1024), torch.rand(2, 45, 80))
        with torch.no_grad():
            original = base(*inputs)
            result = model(*inputs)
        torch.testing.assert_close(result["g_cur"], original["g_cur"])
        torch.testing.assert_close(result["g_des"][:, [0, 1, 3]], original["g_des"][:, [0, 1, 3]])
        torch.testing.assert_close(result["g_des"][:, 2], result["g_cur"][:, 2])
        torch.testing.assert_close(result["e_view"], result["g_des"] - result["g_cur"])

    def test_flow_integrates_constant_velocity_from_zero(self):
        model = ActionHead(12, kind="flow", hidden=32, layers=1).eval()
        condition = torch.randn(2, 12)
        states, times = [], []

        def constant_velocity(x, action, time):
            states.append(action.clone())
            times.append(time.clone())
            return torch.full_like(action, 2.0)

        with patch.object(model, "forward", side_effect=constant_velocity):
            result = model.predict(condition, steps=32)
        torch.testing.assert_close(states[0], torch.zeros(2, 8, 4))
        torch.testing.assert_close(result, torch.full((2, 8, 4), 2.0))
        self.assertEqual(len(times), 32)
        torch.testing.assert_close(times[0], torch.full((2,), 0.5 / 32))

    def test_flow_forward_and_repeatable_sampling(self):
        model = ActionHead(12, kind="flow", hidden=32, layers=1).eval()
        condition = torch.randn(2, 12)
        velocity = model(condition, torch.randn(2, 8, 4), torch.rand(2))
        velocity.square().mean().backward()
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())
        first = model.predict(condition, steps=32)
        second = model.predict(condition, steps=32)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(first).all())
        with self.assertRaises(ValueError):
            model.predict(condition, steps=0)


if __name__ == "__main__":
    unittest.main()
