"""Small, testable helpers for the task-conditioned OMOMO box-carry task.

The functions here are deliberately simulator-independent.  Quaternions use
Isaac Gym/PyTorch3D ``xyzw`` ordering; all returned offsets are world-frame
vectors derived from the reset robot heading, never a hard-coded world axis.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch


SEMANTIC_ROLE_ORDER = (
    "start", "approach", "contact", "lift", "carry_mid", "arrive", "place", "release",
)
_BOX_ALPHA = (0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0)
_BOX_COEFFICIENT = (0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def resolve_semantic_keyframes(
    event_names: Sequence[str],
    event_trigger_frames: Sequence[int] | torch.Tensor,
    semantic_roles: Mapping[str, str],
    fps: float,
) -> tuple[dict[str, int], dict[str, float]]:
    """Resolve configured semantic role names without relying on event order."""
    if fps <= 0.0:
        raise ValueError(f"Motion FPS must be positive, got {fps}")
    if len(event_names) != len(event_trigger_frames):
        raise ValueError(
            "event_names and event_trigger_frames must have identical lengths; "
            f"got {len(event_names)} and {len(event_trigger_frames)}"
        )
    if len(set(event_names)) != len(event_names):
        raise ValueError(f"Semantic event names must be unique, got {list(event_names)}")
    missing_roles = [role for role in SEMANTIC_ROLE_ORDER if role not in semantic_roles]
    if missing_roles:
        raise ValueError(f"semantic_keyframes is missing roles: {missing_roles}")
    name_to_frame = {str(name): int(frame) for name, frame in zip(event_names, event_trigger_frames)}
    resolved: dict[str, int] = {}
    for role in SEMANTIC_ROLE_ORDER:
        event_name = str(semantic_roles[role])
        if event_name not in name_to_frame:
            raise ValueError(
                f"Semantic role {role!r} maps to missing event {event_name!r}; "
                f"available events: {list(event_names)}"
            )
        resolved[role] = name_to_frame[event_name]
    frames = [resolved[role] for role in SEMANTIC_ROLE_ORDER]
    if any(next_frame <= frame for frame, next_frame in zip(frames[:-1], frames[1:])):
        raise ValueError(f"Semantic event frames must be strictly increasing, got {resolved}")
    return resolved, {role: resolved[role] / float(fps) for role in SEMANTIC_ROLE_ORDER}


def smooth_box_carry_offset(
    motion_time: torch.Tensor,
    role_times: Mapping[str, float],
    delta_box: torch.Tensor,
    delta_carry: torch.Tensor,
) -> torch.Tensor:
    """Return a smooth scalar sparse-global offset for every environment.

    The piecewise smoothstep interpolation is only a *global target* edit;
    it never affects local pose/DoF targets.  At the configured role frames it
    evaluates exactly to the requested offsets in the task specification.
    """
    if motion_time.ndim != delta_box.ndim or motion_time.shape != delta_box.shape or motion_time.shape != delta_carry.shape:
        raise ValueError("motion_time, delta_box and delta_carry must have matching shape (num_envs,)")
    knots = torch.tensor([float(role_times[role]) for role in SEMANTIC_ROLE_ORDER], device=motion_time.device)
    values = torch.stack([
        delta_box * float(box_coeff) + delta_carry * float(carry_alpha)
        for box_coeff, carry_alpha in zip(_BOX_COEFFICIENT, _BOX_ALPHA)
    ], dim=1)
    result = torch.zeros_like(motion_time)
    for index in range(len(SEMANTIC_ROLE_ORDER) - 1):
        left, right = knots[index], knots[index + 1]
        active = (motion_time >= left) & (motion_time < right)
        ratio = ((motion_time - left) / (right - left)).clamp(0.0, 1.0)
        smoothstep = ratio * ratio * (3.0 - 2.0 * ratio)
        candidate = values[:, index] + (values[:, index + 1] - values[:, index]) * smoothstep
        result = torch.where(active, candidate, result)
    return torch.where(motion_time >= knots[-1], values[:, -1], result)
