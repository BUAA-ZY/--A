# -*- coding: utf-8 -*-
"""Lightweight simulator for Problem A physics and obscuration evaluation."""
from typing import Callable, List, Optional, Tuple

from .constants import CLOUD_EFFECTIVE_SECONDS, CLOUD_RADIUS, CLOUD_SINK_RATE, DEFAULT_DT
from .dynamics import missile_state_at, uav_state_at, bomb_state_after_release, cloud_center_at
from .occlusion import is_obscured_by_sphere, sample_target_points
from .types import BombEvent, CylinderTarget, SimulationResult, Vec3, ObscurationInterval


def simulate_single_bomb(
    m0: Vec3,
    u0: Vec3,
    heading_to_xy: Vec3,
    u_speed: float,
    bomb: BombEvent,
    true_target: CylinderTarget,
    t_max: float,
    dt: float = DEFAULT_DT,
) -> SimulationResult:
    """Simulate one missile, one UAV, one bomb, evaluate obscuration intervals.

    - Missile flies to origin at constant speed.
    - UAV flies level to heading_to_xy at u_speed.
    - Bomb is released from UAV position/velocity at t_release and detonates at t_detonate.
    """
    t = 0.0
    detonation_position: Optional[Vec3] = None
    cloud_track: List[Tuple[float, Vec3]] = []
    obscured_times: List[bool] = []
    times: List[float] = []

    # Pre-sample target points for union test
    tgt_samples = sample_target_points(true_target, num_side=16, include_top=True)

    # Bomb dynamic state (set upon release)
    bomb_pos: Optional[Vec3] = None
    bomb_vel: Optional[Vec3] = None

    while t <= t_max + 1e-9:
        (m_pos, _m_vel) = missile_state_at(t, m0)
        (u_pos, u_vel) = uav_state_at(t, u0, heading_to_xy, u_speed)

        # Handle bomb release
        if bomb_pos is None and t >= bomb.t_release:
            bomb_pos = (u_pos[0], u_pos[1], u_pos[2])
            bomb_vel = (u_vel[0], u_vel[1], 0.0)

        # Propagate bomb before detonation
        if bomb_pos is not None and detonation_position is None:
            if t + dt >= bomb.t_detonate:
                # Advance to exact detonation time fractionally
                dt_partial = max(0.0, bomb.t_detonate - t)
                if dt_partial > 1e-9:
                    bomb_pos, bomb_vel = bomb_state_after_release(dt_partial, bomb_pos, bomb_vel)  # type: ignore[arg-type]
                    t += dt_partial
                    (m_pos, _m_vel) = missile_state_at(t, m0)
                    (u_pos, u_vel) = uav_state_at(t, u0, heading_to_xy, u_speed)
                detonation_position = bomb_pos
            else:
                bomb_pos, bomb_vel = bomb_state_after_release(dt, bomb_pos, bomb_vel)  # type: ignore[arg-type]

        # Cloud position and obscuration check
        cloud_center: Optional[Vec3] = None
        if detonation_position is not None:
            cloud_center = cloud_center_at(t, detonation_position, bomb.t_detonate, CLOUD_SINK_RATE)
            age = t - bomb.t_detonate
            if 0.0 <= age <= CLOUD_EFFECTIVE_SECONDS:
                # Union over target samples
                is_occ = False
                for tp in tgt_samples:
                    if is_obscured_by_sphere(m_pos, tp, cloud_center, CLOUD_RADIUS):
                        is_occ = True
                        break
                obscured_times.append(is_occ)
            else:
                obscured_times.append(False)
            cloud_track.append((t, cloud_center))
        else:
            obscured_times.append(False)

        times.append(t)
        t += dt

    # Build intervals from boolean mask
    intervals: List[ObscurationInterval] = []
    in_seg = False
    seg_start = 0.0
    for i in range(len(times)):
        t_i = times[i]
        occ = obscured_times[i]
        if occ and not in_seg:
            in_seg = True
            seg_start = t_i
        elif not occ and in_seg:
            in_seg = False
            intervals.append(ObscurationInterval(seg_start, t_i))
    if in_seg:
        intervals.append(ObscurationInterval(seg_start, times[-1]))

    total = sum(max(0.0, iv.t1 - iv.t0) for iv in intervals)
    return SimulationResult(total_time=total, obscured_intervals=intervals, cloud_track=cloud_track)


