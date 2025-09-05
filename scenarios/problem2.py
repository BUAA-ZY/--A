# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict

from physics.types import BombEvent, CylinderTarget
from .base import Scenario


class Problem2(Scenario):
    def __init__(self):
        super().__init__(
            name="Problem2",
            description="FY1单弹干扰M1，优化FY1航向、速度、投放与起爆",
        )

    def initial_states(self) -> Dict:
        # Defaults; algorithms will vary u_speed, heading and timing.
        m0 = (20000.0, 0.0, 2000.0)
        u0 = (17800.0, 0.0, 1800.0)
        heading_to = (0.0, 0.0, 0.0)
        u_speed = 100.0
        bomb = BombEvent(t_release=1.5, t_detonate=1.5 + 3.6)
        true_target = CylinderTarget(center=(0.0, 200.0, 0.0), radius=7.0, height=10.0)
        t_max = 70.0
        return dict(m0=m0, u0=u0, heading_to_xy=heading_to, u_speed=u_speed, bomb=bomb, true_target=true_target, t_max=t_max)


