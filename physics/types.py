# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List, Optional, Tuple

Vec3 = Tuple[float, float, float]


@dataclass
class CylinderTarget:
    center: Vec3  # base center (x, y, z)
    radius: float
    height: float


@dataclass
class MissileState:
    position: Vec3
    velocity: Vec3


@dataclass
class UAVState:
    position: Vec3
    velocity: Vec3  # level flight: z component ~ 0


@dataclass
class BombEvent:
    t_release: float
    t_detonate: float


@dataclass
class CloudState:
    center: Vec3
    radius: float
    t_detonate: float


@dataclass
class ObscurationInterval:
    t0: float
    t1: float


@dataclass
class SimulationResult:
    total_time: float
    obscured_intervals: List[ObscurationInterval]
    cloud_track: List[Tuple[float, Vec3]]  # (t, center)


