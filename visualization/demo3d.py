# -*- coding: utf-8 -*-
"""Minimal 3D visualization demo using matplotlib.

Run as module (recommended):
    python -m visualization.demo3d

Or run as script:
    python visualization/demo3d.py
"""
import math
import time
from typing import Dict, List

try:
    from .renderer3d import Renderer3D
except Exception:
    from renderer3d import Renderer3D


def make_dummy_state(t: float) -> Dict:
    entities: List[Dict] = []
    # Blue aircraft
    entities.append({
        "id": "blue",
        "x": 350 + 120 * math.cos(t),
        "y": 350 + 120 * math.sin(t),
        "z": 200 + 60 * math.sin(0.5 * t),
        "theta": t + math.pi / 2,  # 沿圆周切向方向
        "color": 'b',
        "size": 60,
    })
    # Four green aircrafts
    for k in range(4):
        ang = t + k * math.pi / 2
        entities.append({
            "id": f"g{k}",
            "x": 350 + 150 * math.cos(ang),
            "y": 350 + 150 * math.sin(ang),
            "z": 180 + 40 * math.cos(0.5 * ang),
            "theta": ang + math.pi / 2,
            "color": 'g',
            "size": 50,
        })
    obstacles = [{"x": 350, "y": 420, "z": 150, "radius": 40}]
    return {"entities": entities, "obstacles": obstacles, "axes_origin": (0, 0, 0)}


def main():
    r3d = Renderer3D(title='UAV 3D 可视化', viewsize=(1000, 800, 600))
    r3d.init()
    r3d.set_refresh_rate_hz(60)
    t = 0.0
    while True:
        r3d.draw_scene(make_dummy_state(t))
        t += 0.05


if __name__ == '__main__':
    main()


