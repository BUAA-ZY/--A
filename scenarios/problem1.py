# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict

from physics.constants import MISSILE_SPEED
from physics.types import BombEvent, CylinderTarget
from .base import Scenario


class Problem1(Scenario):
    def __init__(self):
        super().__init__(
            name="Problem1",
            description="单机(FY1)单弹干扰M1，FY1速度120m/s，t_release=1.5s，Δ=3.6s",
        )

    def initial_states(self) -> Dict:
        m0 = (20000.0, 0.0, 2000.0)
        u0 = (17800.0, 0.0, 1800.0)
        heading_to = (0.0, 0.0, 0.0)
        u_speed = 120.0
        bomb = BombEvent(t_release=1.5, t_detonate=1.5 + 3.6)
        true_target = CylinderTarget(center=(0.0, 200.0, 0.0), radius=7.0, height=10.0)
        # Upper bound of time: missile arrival to origin
        t_max = ((m0[0] ** 2 + m0[1] ** 2 + m0[2] ** 2) ** 0.5) / MISSILE_SPEED
        return dict(m0=m0, u0=u0, heading_to_xy=heading_to, u_speed=u_speed, bomb=bomb, true_target=true_target, t_max=t_max)


