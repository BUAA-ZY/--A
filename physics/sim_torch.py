# -*- coding: utf-8 -*-
"""Torch-based vectorized simulators for batch evaluation (GPU-friendly).

We vectorize across candidates in a batch. Time is discretized with dt.
This module provides two entry points:
- eval_problem2_batch_torch
- eval_problem3_batch_torch

Inputs are plain Python floats; tensors are created on the selected device.
"""
from __future__ import annotations
from typing import List, Sequence, Tuple
import math

import torch

from .constants import G_ACCEL, MISSILE_SPEED, CLOUD_SINK_RATE, CLOUD_RADIUS, CLOUD_EFFECTIVE_SECONDS


def _missile_pos_t(t: torch.Tensor, m0: torch.Tensor) -> torch.Tensor:
    # t: [T], m0: [3]
    dir_vec = (torch.tensor([0.0, 0.0, 0.0], device=m0.device, dtype=m0.dtype) - m0)
    dir_unit = dir_vec / (torch.norm(dir_vec) + 1e-9)
    vel = dir_unit * MISSILE_SPEED
    return m0 + t[:, None] * vel[None, :]


def _uav_state_t(t: torch.Tensor, u0: torch.Tensor, heading_to: torch.Tensor, speed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # t: [T] or [B,], u0: [3], heading_to: [3], speed: [] or [B]
    dir_xy = heading_to - u0
    dir_xy = torch.tensor([dir_xy[0], dir_xy[1], 0.0], device=u0.device, dtype=u0.dtype)
    dir_unit = dir_xy / (torch.norm(dir_xy) + 1e-9)
    vel = dir_unit * speed
    # Broadcast to time
    if t.dim() == 1:
        pos = u0 + t[:, None] * vel[None, :]
        pos[:, 2] = u0[2]
    else:
        # t is [B], speed [B]
        pos = u0 + t[:, None] * vel
        pos[:, 2] = u0[2]
    return pos, vel


def _bomb_detonation_position(u_pos_at_release: torch.Tensor, u_vel_xy: torch.Tensor, dt_delay: torch.Tensor) -> torch.Tensor:
    # After release, bomb follows projectile motion until detonation time delay
    # u_pos_at_release: [B,3], u_vel_xy: [B,3] (z=0), dt_delay: [B]
    p = u_pos_at_release.clone()
    v = u_vel_xy.clone()
    # Advance for dt_delay (closed form)
    p = p + v * dt_delay[:, None]
    p[:, 2] = p[:, 2] - 0.5 * G_ACCEL * (dt_delay ** 2)
    return p


def _cloud_center_after(t: torch.Tensor, det_pos: torch.Tensor, t_e: torch.Tensor) -> torch.Tensor:
    # t: [T], det_pos: [B,3], t_e: [B]
    # Cloud sinks at constant rate after t_e
    dt = (t[None, :] - t_e[:, None]).clamp(min=0.0)
    z = (det_pos[:, 2:3] - CLOUD_SINK_RATE * dt).clamp(min=0.0)
    xy = det_pos[:, :2]
    return torch.cat([xy[:, None, :].expand(-1, t.numel(), -1), z], dim=-1)


def _distance_point_to_segment(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # p: [B, T, 3]; a: [T, 3]; b: [T, 3]
    ab = (b - a)[None, :, :]
    ap = p - a[None, :, :]
    ab2 = (ab * ab).sum(dim=-1).clamp_min(1e-9)
    t = ((ap * ab).sum(dim=-1) / ab2).clamp(0.0, 1.0)
    proj = a[None, :, :] + ab * t[..., None]
    d = torch.norm(p - proj, dim=-1)
    return d


def _compute_obscuration_mask(m_line: torch.Tensor, tgt: torch.Tensor, cloud_centers: torch.Tensor) -> torch.Tensor:
    # m_line: [T,3]; tgt: [3]; cloud_centers: [B, T, 3]
    # Return mask: [B, T] indicating d<=R
    p = cloud_centers  # [B,T,3]
    a = m_line  # [T,3]
    b = tgt  # [3]
    b_line = b[None, :].expand(m_line.size(0), -1)  # [T,3]
    d = _distance_point_to_segment(p, a, b_line)
    return d <= CLOUD_RADIUS


def eval_problem2_batch_torch(
    xs: Sequence[Sequence[float]],
    m0: Sequence[float],
    u0: Sequence[float],
    true_target_center: Sequence[float],
    t_max: float,
    dt: float,
    device: str = 'cuda',
) -> List[float]:
    # xs: [B, 4] = [theta, u_speed, t_release, det_delay]
    B = len(xs)
    if B == 0:
        return []
    dev = torch.device(device)
    dtype = torch.float32
    # Time axis
    T = int(math.ceil(t_max / dt)) + 1
    t = torch.linspace(0.0, t_max, T, device=dev, dtype=dtype)
    # Constants
    m0_t = torch.tensor(m0, device=dev, dtype=dtype)
    u0_t = torch.tensor(u0, device=dev, dtype=dtype)
    tgt = torch.tensor(true_target_center, device=dev, dtype=dtype)
    # Missile line (same across batch)
    m_line = _missile_pos_t(t, m0_t)
    # Batch parameters
    theta = torch.tensor([x[0] for x in xs], device=dev, dtype=dtype)
    u_speed = torch.tensor([x[1] for x in xs], device=dev, dtype=dtype)
    t_r = torch.tensor([x[2] for x in xs], device=dev, dtype=dtype)
    dly = torch.tensor([x[3] for x in xs], device=dev, dtype=dtype)
    # Heading to from theta
    heading_to = torch.stack([u0_t[0] + torch.cos(theta), u0_t[1] + torch.sin(theta), torch.zeros_like(theta)], dim=-1)
    # UAV position and velocity at release
    u_pos_r, u_vel = _uav_state_t(t_r, u0_t, heading_to, u_speed)
    # Detonation moment and position
    t_e = t_r + dly
    det_pos = _bomb_detonation_position(u_pos_r, u_vel, dly)
    # Cloud centers along time [B,T,3]
    cloud_centers = _cloud_center_after(t, det_pos, t_e)
    # Obscuration mask, only valid for age within 20s
    age = (t[None, :] - t_e[:, None])
    valid = (age >= 0.0) & (age <= CLOUD_EFFECTIVE_SECONDS)
    mask = _compute_obscuration_mask(m_line, tgt, cloud_centers) & valid
    # Integrate over time
    total = (mask.float() * dt).sum(dim=1)
    return total.detach().cpu().tolist()


def eval_problem3_batch_torch(
    xs: Sequence[Sequence[float]],
    m0: Sequence[float],
    u0: Sequence[float],
    true_target_center: Sequence[float],
    t_max: float,
    dt: float,
    device: str = 'cuda',
) -> List[float]:
    # xs: [B, 8] = [theta, u, t1,d1, t2,d2, t3,d3]
    if len(xs) == 0:
        return []
    dev = torch.device(device)
    dtype = torch.float32
    T = int(math.ceil(t_max / dt)) + 1
    t = torch.linspace(0.0, t_max, T, device=dev, dtype=dtype)

    m0_t = torch.tensor(m0, device=dev, dtype=dtype)
    u0_t = torch.tensor(u0, device=dev, dtype=dtype)
    tgt = torch.tensor(true_target_center, device=dev, dtype=dtype)
    m_line = _missile_pos_t(t, m0_t)

    X = torch.tensor(xs, device=dev, dtype=dtype)
    theta = X[:, 0]
    u_speed = X[:, 1]
    t_r = X[:, [2, 4, 6]]
    dly = X[:, [3, 5, 7]]

    # enforce spacing >=1s softly (clip)
    t_r_sorted, _ = torch.sort(t_r, dim=1)
    t_r_fixed = t_r_sorted.clone()
    for i in range(2):
        gap = t_r_fixed[:, i+1] - t_r_fixed[:, i]
        need = (1.0 - gap).clamp(min=0.0)
        t_r_fixed[:, i+1] = t_r_fixed[:, i+1] + need
    t_r = t_r_fixed

    heading_to = torch.stack([u0_t[0] + torch.cos(theta), u0_t[1] + torch.sin(theta), torch.zeros_like(theta)], dim=-1)
    # For each bomb i=0..2, compute detonation pos and clouds
    total = torch.zeros(X.size(0), device=dev, dtype=dtype)
    dt_tensor = torch.tensor(dt, device=dev, dtype=dtype)
    union_mask = torch.zeros((X.size(0), T), device=dev, dtype=torch.bool)
    for i in range(3):
        t_ri = t_r[:, i]
        dly_i = dly[:, i]
        u_pos_r, u_vel = _uav_state_t(t_ri, u0_t, heading_to, u_speed)
        t_e = t_ri + dly_i
        det_pos = _bomb_detonation_position(u_pos_r, u_vel, dly_i)
        cloud_centers = _cloud_center_after(t, det_pos, t_e)
        age = (t[None, :] - t_e[:, None])
        valid = (age >= 0.0) & (age <= CLOUD_EFFECTIVE_SECONDS)
        mask_i = _compute_obscuration_mask(m_line, tgt, cloud_centers) & valid
        union_mask = union_mask | mask_i
    total = (union_mask.float() * dt).sum(dim=1)
    return total.detach().cpu().tolist()
