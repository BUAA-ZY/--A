# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class TrainConfig:
    scenario: str = "problem1"  # problem1..problem5
    optimizer: str = "random_search"
    steps: int = 200
    log_dir: str = "runs"
    seed: int = 42
    # 并行评估：最大并行进程数（<=1 表示禁用并行）
    max_workers: int = 0
    # GPU 配置
    use_gpu: bool = True
    gpu_device: str = 'cuda'
    dt: float = 0.02
    # 批并行/规模（GPU/CPU通用）
    rs_batch_size: int = 32
    pso_swarm_size: int = 24
    ga_pop_size: int = 40
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
    # Problem 3: FY1三弹（统一航向/速度），决策变量为 [theta, u_speed, t_release1, det_delay1, t_release2, det_delay2, t_release3, det_delay3]
    # 可直接复用 Problem2 的范围；如需更紧的约束可再细化。


