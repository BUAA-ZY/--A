# -*- coding: utf-8 -*-
"""Scenario visualization for the 2025 MCM/ICM A-Problem.

Run as module (recommended):
    python -m visualization.scenario

This script visualizes:
  - Missiles M1, M2, M3 flying straight toward the fake target (origin)
  - UAVs FY1..FY5 flying level (constant altitude) along set headings/speeds
  - A true cylindrical target at (0, 200, 0) with radius 7 m, height 10 m
  - A smoke cloud sphere (radius 10 m) created by a bomb detonation that sinks at 3 m/s

Default configuration reproduces Problem 1 conditions.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import time

try:
    from .renderer3d import Renderer3D
except Exception:
    from renderer3d import Renderer3D


# -----------------------------
# Data models
# -----------------------------

Vec3 = Tuple[float, float, float]


def _norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _unit(v: Vec3) -> Vec3:
    n = _norm(v)
    if n <= 1e-9:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def _add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


@dataclass
class Missile:
    name: str
    position: Vec3
    target: Vec3 = (0.0, 0.0, 0.0)
    speed: float = 300.0  # m/s
    color: str = 'r'
    size: float = 60.0

    def step(self, dt: float):
        direction = _unit((self.target[0] - self.position[0],
                           self.target[1] - self.position[1],
                           self.target[2] - self.position[2]))
        self.position = _add(self.position, _scale(direction, self.speed * dt))

    def to_entity(self) -> Dict:
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        theta = math.atan2(dy, dx)
        return {"id": self.name, "x": self.position[0], "y": self.position[1], "z": self.position[2], "theta": theta, "color": self.color, "size": self.size}


@dataclass
class UAV:
    name: str
    position: Vec3
    heading_xy_to: Vec3  # a point in XY plane to look at (z ignored)
    speed: float  # 70..140 m/s
    color: str = 'b'
    size: float = 50.0

    def __post_init__(self):
        # Level flight: heading in XY plane only, altitude constant
        dir_xy = _unit((self.heading_xy_to[0] - self.position[0],
                        self.heading_xy_to[1] - self.position[1],
                        0.0))
        self._vel = _scale(dir_xy, self.speed)

    def step(self, dt: float):
        self.position = (self.position[0] + self._vel[0] * dt,
                         self.position[1] + self._vel[1] * dt,
                         self.position[2])

    def to_entity(self) -> Dict:
        theta = math.atan2(self._vel[1], self._vel[0]) if abs(self._vel[0]) + abs(self._vel[1]) > 1e-9 else 0.0
        return {"id": self.name, "x": self.position[0], "y": self.position[1], "z": self.position[2], "theta": theta, "color": self.color, "size": self.size}


@dataclass
class Bomb:
    name: str
    position: Vec3
    velocity: Vec3
    t_release: float
    t_detonate: float
    g: float = 9.8
    detonated: bool = False
    t_detonated: Optional[float] = None

    def step(self, t: float, dt: float):
        if self.detonated or t < self.t_release:
            return
        if t >= self.t_detonate and not self.detonated:
            self.detonated = True
            self.t_detonated = t
            return
        # Projectile motion after release
        # z-velocity initially equals release vertical velocity (here 0 for level flight)
        self.position = (self.position[0] + self.velocity[0] * dt,
                         self.position[1] + self.velocity[1] * dt,
                         self.position[2] + self.velocity[2] * dt - 0.5 * self.g * dt * dt)
        # Update vertical velocity due to gravity
        self.velocity = (self.velocity[0], self.velocity[1], self.velocity[2] - self.g * dt)

    def cloud_state(self, t: float) -> Optional[Dict]:
        if not self.detonated or self.t_detonated is None:
            return None
        # Cloud sinks at 3 m/s after detonation, radius 10 m, effective for 20 s
        age = max(0.0, t - self.t_detonated)
        center_z = max(0.0, self.position[2] - 3.0 * age)
        alpha = 0.35 if age <= 20.0 else max(0.05, 0.35 * max(0.0, 1.0 - (age - 20.0) / 10.0))
        return {"x": self.position[0], "y": self.position[1], "z": center_z, "radius": 10.0, "color": 'gray', "alpha": alpha}


# -----------------------------
# Scenario setup (Problem 1 default)
# -----------------------------

def build_problem1() -> Dict:
    # Missiles
    missiles = [
        Missile("M1", (20000.0, 0.0, 2000.0), target=(0.0, 0.0, 0.0), color='r'),
        Missile("M2", (19000.0, 600.0, 2100.0), target=(0.0, 0.0, 0.0), color='r'),
        Missile("M3", (18000.0, -600.0, 1900.0), target=(0.0, 0.0, 0.0), color='r'),
    ]
    # UAVs (Problem 1: only FY1 used, heading toward fake target at origin, speed 120 m/s)
    uavs = [
        UAV("FY1", (17800.0, 0.0, 1800.0), heading_xy_to=(0.0, 0.0, 0.0), speed=120.0, color='b'),
        UAV("FY2", (12000.0, 1400.0, 1400.0), heading_xy_to=(0.0, 0.0, 0.0), speed=90.0, color='b'),
        UAV("FY3", (6000.0, -3000.0, 700.0), heading_xy_to=(0.0, 0.0, 0.0), speed=90.0, color='b'),
        UAV("FY4", (11000.0, 2000.0, 1800.0), heading_xy_to=(0.0, 0.0, 0.0), speed=90.0, color='b'),
        UAV("FY5", (13000.0, -2000.0, 1300.0), heading_xy_to=(0.0, 0.0, 0.0), speed=90.0, color='b'),
    ]

    # Bomb for Problem 1: FY1 releases at t=1.5s, detonates 3.6s later
    # Release position and velocity will be set at runtime from FY1 state
    bomb_cfg = {"name": "B1", "t_release": 1.5, "t_detonate": 1.5 + 3.6}

    world = {
        "missiles": missiles,
        "uavs": uavs,
        "bomb_cfg": bomb_cfg,
        "true_target": {"x": 0.0, "y": 200.0, "z": 0.0, "radius": 7.0, "height": 10.0},
        "fake_target": (0.0, 0.0, 0.0),
        # Bounds chosen to cover [x in -1000..20000, y in -4000..4000, z in 0..2500]
        "bounds": ((-1000.0, 20000.0), (-4000.0, 4000.0), (0.0, 2500.0)),
    }
    return world


# -----------------------------
# Simulation & rendering
# -----------------------------

def run(world: Dict, refresh_hz: float = 60.0):
    r3d = Renderer3D(title='A题 场景可视化', viewsize=(1000, 800, 600))
    r3d.init()
    r3d.set_refresh_rate_hz(refresh_hz)
    r3d.enable_interaction()
    # Set custom world bounds
    r3d.set_bounds(world["bounds"][0], world["bounds"][1], world["bounds"][2])

    missiles: List[Missile] = world["missiles"]
    uavs: List[UAV] = world["uavs"]
    true_target = world["true_target"]
    fake_target = world["fake_target"]

    # Bomb is created lazily at t_release using FY1 state
    bomb_cfg = world.get("bomb_cfg") or {}
    bomb: Optional[Bomb] = None

    t0 = time.monotonic()
    last = t0
    while True:
        now = time.monotonic()
        t = now - t0
        dt = max(1e-3, now - last)
        last = now

        # Step dynamics
        for m in missiles:
            m.step(dt)
        for u in uavs:
            u.step(dt)

        # Create bomb at t_release if not created
        if bomb is None and t >= bomb_cfg.get("t_release", 1e9):
            fy1 = next(u for u in uavs if u.name == "FY1")
            # Bomb initial position equals FY1 position at release
            p0 = fy1.position
            # Bomb initial velocity equals FY1 horizontal velocity, vertical 0 (level)
            # Use FY1 current horizontal velocity components directly
            vx, vy = fy1._vel[0], fy1._vel[1]
            bomb = Bomb(
                name=bomb_cfg.get("name", "B1"),
                position=(p0[0], p0[1], p0[2]),
                velocity=(vx, vy, 0.0),
                t_release=bomb_cfg.get("t_release", t),
                t_detonate=bomb_cfg.get("t_detonate", t + 3.6),
            )

        # Step bomb
        if bomb is not None:
            bomb.step(t, dt)

        # Prepare draw dict
        entities = []
        entities.extend([m.to_entity() for m in missiles])
        entities.extend([u.to_entity() for u in uavs])
        spheres = []
        if bomb is not None:
            cloud = bomb.cloud_state(t)
            if cloud is not None:
                spheres.append(cloud)

        cylinders = [{
            "x": true_target["x"],
            "y": true_target["y"],
            "z": true_target["z"],
            "height": true_target["height"],
            "radius": true_target["radius"],
            "color": 'orange',
            "alpha": 0.5,
        }]

        # Fake target marker as small sphere
        spheres.append({"x": fake_target[0], "y": fake_target[1], "z": fake_target[2], "radius": 5.0, "color": 'y', "alpha": 0.8})

        r3d.draw_scene({
            "entities": entities,
            "spheres": spheres,
            "cylinders": cylinders,
            "axes_origin": (0.0, 0.0, 0.0),
        })


def main():
    world = build_problem1()
    run(world, refresh_hz=60.0)


if __name__ == '__main__':
    main()


