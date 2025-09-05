# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

from physics.types import CylinderTarget, BombEvent, Vec3


@dataclass
class Scenario:
    name: str
    description: str

    def initial_states(self) -> Dict:
        """Return dict with keys: m0, u0, heading_to_xy, u_speed, bomb, true_target, t_max."""
        raise NotImplementedError


