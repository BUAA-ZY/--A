# -*- coding: utf-8 -*-
"""Deterministic kinematics for missiles, UAVs, bombs, and smoke clouds."""
from typing import Tuple

from .constants import G_ACCEL, MISSILE_SPEED
from .geometry import unit, add, scale
from .types import Vec3


def missile_state_at(t: float, m0: Vec3, target: Vec3 = (0.0, 0.0, 0.0)) -> Tuple[Vec3, Vec3]:
    """Straight-line constant-speed to target."""
    direction = unit((target[0] - m0[0], target[1] - m0[1], target[2] - m0[2]))
    velocity = scale(direction, MISSILE_SPEED)
    position = add(m0, scale(velocity, t))
    return position, velocity


def uav_state_at(t: float, f0: Vec3, heading_to_xy: Vec3, speed: float) -> Tuple[Vec3, Vec3]:
    """Level flight: heading defined in XY plane toward heading_to_xy, constant altitude."""
    vxvy = unit((heading_to_xy[0] - f0[0], heading_to_xy[1] - f0[1], 0.0))
    velocity = (vxvy[0] * speed, vxvy[1] * speed, 0.0)
    position = (f0[0] + velocity[0] * t, f0[1] + velocity[1] * t, f0[2])
    return position, velocity


def bomb_state_after_release(dt: float, p0: Vec3, v0: Vec3) -> Tuple[Vec3, Vec3]:
    """Projectile motion over dt after release at (p0,v0)."""
    px = p0[0] + v0[0] * dt
    py = p0[1] + v0[1] * dt
    pz = p0[2] + v0[2] * dt - 0.5 * G_ACCEL * dt * dt
    vx = v0[0]
    vy = v0[1]
    vz = v0[2] - G_ACCEL * dt
    return (px, py, pz), (vx, vy, vz)


def cloud_center_at(t: float, detonation_position: Vec3, t_detonate: float, sink_rate: float) -> Vec3:
    if t <= t_detonate:
        return detonation_position
    dz = max(0.0, detonation_position[2] - sink_rate * (t - t_detonate))
    return (detonation_position[0], detonation_position[1], dz)


