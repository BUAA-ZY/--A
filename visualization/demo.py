# -*- coding: utf-8 -*-
"""Minimal standalone visualization demo.

Run as module (recommended):
    python -m visualization.demo

Or run as script:
    python visualization/demo.py
"""
import math
import time
from typing import List, Dict

try:
    # When run as module: python -m visualization.demo
    from .renderer import Renderer
except Exception:
    # When run as a script: python visualization/demo.py
    from renderer import Renderer


def make_dummy_state(t: float) -> Dict:
    # Circular motion for hero; enemies follow offset
    hero_x = 350 + 100 * math.cos(t)
    hero_y = 350 + 100 * math.sin(t)

    enemies: List[Dict] = []
    for k in range(4):
        ang = t + k * math.pi / 2
        enemies.append({
            "x": 350 + 150 * math.cos(ang),
            "y": 350 + 150 * math.sin(ang),
            "theta": ang,
        })

    return {
        "hero": [{"x": hero_x, "y": hero_y, "theta": t}],
        "enemies": enemies,
        "goal": {"x": 600, "y": 200},
        "obstacle": {"x": 350, "y": 420},
        "trajectory": [(hero_x - 2 * i, hero_y - 2 * i) for i in range(30)],
        "enemy_trajectories": [
            [(e["x"] - i, e["y"] - i) for i in range(15)] for e in enemies
        ],
        "metrics": {"t": round(t, 2)},
    }


def main():
    rnd = Renderer(window_title="UAV 可视化独立演示")
    rnd.init()
    running = True
    t = 0.0
    while running:
        running = rnd.poll_events()
        rnd.begin_frame()
        rnd.draw_scene(make_dummy_state(t))
        rnd.end_frame()
        t += 0.05


if __name__ == "__main__":
    main()


