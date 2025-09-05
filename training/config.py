# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class TrainConfig:
    scenario: str = "problem1"  # problem1..problem5
    optimizer: str = "random_search"
    steps: int = 200
    log_dir: str = "runs"
    seed: int = 42
    # Search bounds for Problem 1 style single-bomb example
    # x = [t_release, det_delay(=t_detonate-t_release), u_speed]
    t_release_min: float = 0.5
    t_release_max: float = 5.0
    det_delay_min: float = 1.0
    det_delay_max: float = 6.0
    u_speed_min: float = 70.0
    u_speed_max: float = 140.0
    # Problem 2 adds heading angle theta (rad)
    heading_min: float = -3.141592653589793
    heading_max: float = 3.141592653589793


