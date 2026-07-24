"""Unit tests for semantic role resolution and task-conditioned offsets."""
import unittest

import isaacgym  # Isaac Gym must precede torch in this environment.
import torch

from legged_gym.envs.base.box_carry import (
    SEMANTIC_ROLE_ORDER,
    resolve_semantic_keyframes,
    smooth_box_carry_offset,
)


class BoxCarryKeyframeTest(unittest.TestCase):
    def setUp(self):
        self.names = list(SEMANTIC_ROLE_ORDER)
        self.frames = torch.arange(0, 80, 10)
        self.roles = {name: name for name in self.names}
        _, self.times = resolve_semantic_keyframes(self.names, self.frames, self.roles, fps=10.0)

    def test_roles_are_name_resolved_and_ordered(self):
        frames, times = resolve_semantic_keyframes(self.names, self.frames, self.roles, fps=10.0)
        self.assertEqual(frames["lift"], 30)
        self.assertAlmostEqual(times["release"], 7.0)
        with self.assertRaisesRegex(ValueError, "missing event"):
            resolve_semantic_keyframes(self.names, self.frames, {**self.roles, "lift": "wrong"}, fps=10.0)

    def test_fixed_demo_degenerates_to_original_reference(self):
        motion_time = torch.tensor([0.0, 1.0, 3.0, 7.0])
        result = smooth_box_carry_offset(motion_time, self.times, torch.zeros(4), torch.zeros(4))
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_delta_box_and_carry_affect_expected_roles(self):
        motion_time = torch.tensor([0.0, 1.0, 3.0, 4.0, 5.0, 7.0])
        delta_box = torch.full((6,), 0.2)
        delta_carry = torch.full((6,), 0.4)
        result = smooth_box_carry_offset(motion_time, self.times, delta_box, delta_carry)
        expected = torch.tensor([0.0, 0.2, 0.2, 0.4, 0.6, 0.6])
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
