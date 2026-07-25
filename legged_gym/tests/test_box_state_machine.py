"""Unit tests for the phase-conditioned Box-Carry state machine.

These tests intentionally avoid creating an Isaac Gym simulation.  They build a
minimal ``LeggedRobot`` instance with the tensor fields consumed by
``_update_box_carry_state`` and replace hand geometry queries with deterministic
test tensors.
"""
from types import MethodType, SimpleNamespace
import unittest

import isaacgym  # noqa: F401  # Isaac Gym must precede torch in this environment.
import torch

from legged_gym.envs.base.motion_tracking import LeggedRobot


ROLE_TIMES = {
    "start": 0.0,
    "approach": 1.0,
    "contact": 2.0,
    "lift": 3.0,
    "carry_mid": 4.0,
    "arrive": 5.0,
    "place": 6.0,
    "release": 7.0,
}


def _make_task(**overrides):
    values = dict(
        contact_velocity_tolerance=10.0,
        contact_distance_threshold=0.14,
        contact_penetration_tolerance=0.04,
        require_opposite_contact_sides=True,
        contact_hold_steps=1,
        contact_grace_steps=1,
        lift_threshold=0.08,
        max_tilt=0.65,
        lift_hold_steps=1,
        minimum_transport_height=0.08,
        max_transport_tilt=0.65,
        goal_position_threshold=0.12,
        arrive_hold_steps=2,
        place_height_threshold=0.035,
        place_tilt_threshold=0.65,
        place_linear_velocity_threshold=0.20,
        place_angular_velocity_threshold=1.0,
        place_hold_steps=1,
        release_distance_threshold=0.16,
        release_hold_steps=1,
        drop_height_threshold=0.05,
        drop_hold_steps=1,
        max_progress_per_step=0.05,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_env(motion_time, num_envs=1, task=None):
    env = LeggedRobot.__new__(LeggedRobot)
    env.box_carry_enabled = True
    env.device = torch.device("cpu")
    env.num_envs = num_envs
    env.cfg = SimpleNamespace(box_carry_task=task or _make_task())
    env.semantic_role_times = ROLE_TIMES
    env.motion_time = torch.full((num_envs,), float(motion_time))
    env.box_hand_indices = torch.tensor([0, 1], dtype=torch.long)
    env.rigid_body_states = torch.zeros(num_envs, 2, 13)
    env.box_half_extent = torch.tensor([0.1765, 0.172, 0.1525])
    env.box_pos = torch.zeros(num_envs, 3)
    env.box_pos[:, 2] = env.box_half_extent[2]
    env.box_quat = torch.zeros(num_envs, 4)
    env.box_quat[:, 3] = 1.0
    env.box_lin_vel = torch.zeros(num_envs, 3)
    env.box_ang_vel = torch.zeros(num_envs, 3)
    env.box_goal_pos = torch.zeros(num_envs, 3)
    env.box_ground_z = torch.zeros(num_envs)
    env.box_prev_goal_distance = torch.ones(num_envs)
    env.box_goal_progress = torch.zeros(num_envs)
    env.box_current_goal_distance = torch.ones(num_envs)
    env.left_hand_box_distance = torch.zeros(num_envs)
    env.right_hand_box_distance = torch.zeros(num_envs)
    env.left_hand_box_penetration = torch.zeros(num_envs)
    env.right_hand_box_penetration = torch.zeros(num_envs)
    env.box_hand_contact_side = torch.zeros(num_envs, 2)
    env.box_tilt = torch.zeros(num_envs)
    env.contact_proxy = torch.zeros(num_envs, dtype=torch.bool)
    env.grasp_contact_with_grace = torch.zeros(num_envs, dtype=torch.bool)
    env.valid_transport = torch.zeros(num_envs, dtype=torch.bool)
    env.has_contacted = torch.zeros(num_envs, dtype=torch.bool)
    env.has_lifted = torch.zeros(num_envs, dtype=torch.bool)
    env.has_arrived = torch.zeros(num_envs, dtype=torch.bool)
    env.has_placed = torch.zeros(num_envs, dtype=torch.bool)
    env.has_released = torch.zeros(num_envs, dtype=torch.bool)
    env.box_dropped = torch.zeros(num_envs, dtype=torch.bool)
    env.box_full_task_success = torch.zeros(num_envs, dtype=torch.bool)
    env.box_max_height_after_lift = torch.zeros(num_envs)
    env.box_contact_steps = torch.zeros(num_envs, dtype=torch.long)
    env.box_grasp_grace_steps = torch.zeros(num_envs, dtype=torch.long)
    env.box_lift_steps = torch.zeros(num_envs, dtype=torch.long)
    env.box_arrive_steps = torch.zeros(num_envs, dtype=torch.long)
    env.box_place_steps = torch.zeros(num_envs, dtype=torch.long)
    env.box_release_steps = torch.zeros(num_envs, dtype=torch.long)
    env.box_drop_steps = torch.zeros(num_envs, dtype=torch.long)
    env.box_drop_events = torch.zeros(num_envs, dtype=torch.long)
    env.box_transport_without_contact_steps = torch.zeros(num_envs, dtype=torch.long)
    env.box_transport_progress_steps = torch.zeros(num_envs, dtype=torch.long)
    env.test_distances = torch.full((num_envs, 2), 0.50)
    env.test_penetration = torch.zeros(num_envs, 2)
    env.test_contact_side = torch.tensor([[1.0, -1.0]]).repeat(num_envs, 1)

    def fake_contact_points(self, hand_states):
        return torch.zeros(self.num_envs, 2, 3)

    def fake_side_terms(self, hand_points):
        return self.test_distances, self.test_penetration, self.test_contact_side

    def no_transition_print(self, state_name, transitioned, box_height):
        return None

    env._box_hand_contact_points = MethodType(fake_contact_points, env)
    env._box_side_contact_terms = MethodType(fake_side_terms, env)
    env._log_box_state_transition = MethodType(no_transition_print, env)
    return env


class BoxStateMachineTest(unittest.TestCase):
    def test_approach_phase_cannot_lift_or_arrive(self):
        env = _make_env(motion_time=1.5)
        env.test_distances[:] = 0.01
        env.box_pos[:, 2] = env.box_half_extent[2] + 0.20
        env.box_goal_pos[:, :2] = env.box_pos[:, :2]
        env._update_box_carry_state()
        self.assertTrue(bool(env.has_contacted.item()))
        self.assertFalse(bool(env.has_lifted.item()))
        self.assertFalse(bool(env.has_arrived.item()))

    def test_before_contact_window_cannot_lift(self):
        env = _make_env(motion_time=0.5)
        env.test_distances[:] = 0.01
        env.box_pos[:, 2] = env.box_half_extent[2] + 0.20
        env._update_box_carry_state()
        self.assertFalse(bool(env.has_contacted.item()))
        self.assertFalse(bool(env.has_lifted.item()))

    def test_short_goal_pass_does_not_place(self):
        env = _make_env(motion_time=3.5)
        env.has_lifted[:] = True
        env.box_pos[:, :2] = 0.0
        env.box_goal_pos[:, :2] = 0.0
        env._update_box_carry_state()
        self.assertFalse(bool(env.has_arrived.item()))
        self.assertFalse(bool(env.has_placed.item()))

    def test_thrown_box_gets_no_transport_reward(self):
        env = _make_env(motion_time=3.5)
        env.has_lifted[:] = True
        env.box_pos[:, 0] = 0.5
        env.box_pos[:, 2] = env.box_half_extent[2] + 0.20
        env.box_goal_pos[:, 0] = 1.0
        env.box_prev_goal_distance[:] = 1.0
        env.test_distances[:] = 0.50
        env._update_box_carry_state()
        reward = env._reward_box_transport()
        self.assertEqual(float(reward.item()), 0.0)
        self.assertEqual(int(env.box_transport_without_contact_steps.item()), 1)

    def test_lost_grasp_after_lift_marks_dropped(self):
        env = _make_env(motion_time=3.5)
        env.has_lifted[:] = True
        env.box_max_height_after_lift[:] = 0.20
        env.box_pos[:, 2] = env.box_half_extent[2] + 0.01
        env.test_distances[:] = 0.50
        env._update_box_carry_state()
        self.assertTrue(bool(env.box_dropped.item()))
        self.assertEqual(int(env.box_drop_events.item()), 1)

    def test_normal_place_does_not_drop(self):
        env = _make_env(motion_time=5.5)
        env.has_lifted[:] = True
        env.has_arrived[:] = True
        env.box_max_height_after_lift[:] = 0.20
        env.box_pos[:, 2] = env.box_half_extent[2]
        env.box_goal_pos[:, :2] = env.box_pos[:, :2]
        env.test_distances[:] = 0.50
        env._update_box_carry_state()
        self.assertTrue(bool(env.has_placed.item()))
        self.assertFalse(bool(env.box_dropped.item()))

    def test_success_becomes_false_if_released_box_leaves_goal(self):
        env = _make_env(motion_time=6.5)
        env.has_lifted[:] = True
        env.has_arrived[:] = True
        env.has_placed[:] = True
        env.box_pos[:, 2] = env.box_half_extent[2]
        env.box_goal_pos[:, :2] = env.box_pos[:, :2]
        env.test_distances[:] = 0.50
        env._update_box_carry_state()
        self.assertTrue(bool(env.has_released.item()))
        self.assertTrue(bool(env.box_full_task_success.item()))

        env.box_pos[:, 0] = 10.0
        env._update_box_carry_state()
        self.assertTrue(bool(env.has_released.item()))
        self.assertFalse(bool(env.box_full_task_success.item()))


if __name__ == "__main__":
    unittest.main()
