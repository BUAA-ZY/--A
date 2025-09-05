# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict

from physics.types import BombEvent, CylinderTarget
from .base import Scenario


class Problem5(Scenario):
    def __init__(self):
        super().__init__(
            name="Problem5",
            description="五机多弹干扰M1-M3，占位场景，算法负责调度与资源约束",
        )

    def initial_states(self) -> Dict:
        m0 = (20000.0, 0.0, 2000.0)
        u0 = (17800.0, 0.0, 1800.0)
        heading_to = (0.0, 0.0, 0.0)
        u_speed = 100.0
        bomb = BombEvent(t_release=1.0, t_detonate=1.0 + 3.0)
        true_target = CylinderTarget(center=(0.0, 200.0, 0.0), radius=7.0, height=10.0)
        t_max = 70.0
        return dict(m0=m0, u0=u0, heading_to_xy=heading_to, u_speed=u_speed, bomb=bomb, true_target=true_target, t_max=t_max)


