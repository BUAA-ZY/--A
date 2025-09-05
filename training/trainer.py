# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Sequence
import os
import math

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # TensorBoard optional at import time

from algorithms import get_optimizer
from algorithms.base import SearchSpace
from scenarios import Problem1, Problem2, Problem3, Problem4, Problem5
from physics.simulation import simulate_single_bomb
from physics.types import BombEvent, CylinderTarget
from .config import TrainConfig


def _get_scenario(name: str):
    key = name.lower().strip()
    if key == 'problem1':
        return Problem1()
    if key == 'problem2':
        return Problem2()
    if key == 'problem3':
        return Problem3()
    if key == 'problem4':
        return Problem4()
    if key == 'problem5':
        return Problem5()
    raise KeyError(f"Unknown scenario: {name}")


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.scenario = _get_scenario(cfg.scenario)
        self.OptimizerCls = get_optimizer(cfg.optimizer)
        self.writer = None
        if SummaryWriter is not None:
            run_name = f"{cfg.scenario}_{cfg.optimizer}_seed{cfg.seed}"
            # Robust directory creation: avoid collisions with files
            import tempfile
            import time as _time

            def _ensure_dir(path: str) -> str:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        return path
                    # path exists as a file -> try rename with suffix
                    parent = os.path.dirname(path) or "."
                    base = os.path.basename(path)
                    for suffix in ["_tb", "_tb2", f"_tb_{int(_time.time())}"]:
                        candidate = os.path.join(parent, base + suffix)
                        if not os.path.exists(candidate):
                            os.makedirs(candidate, exist_ok=True)
                            return candidate
                    # fallback to temp dir
                    return tempfile.mkdtemp(prefix="runs_", dir=parent)
                else:
                    os.makedirs(path, exist_ok=True)
                    return path

            try:
                base_dir = _ensure_dir(self.cfg.log_dir)
                run_dir = os.path.join(base_dir, run_name)
                run_dir = _ensure_dir(run_dir)
                self.writer = SummaryWriter(run_dir)
            except Exception:
                # Disable writer but keep training running
                self.writer = None

    def _build_space_problem1(self) -> SearchSpace:
        names = ["t_release", "det_delay", "u_speed"]
        bounds = [
            (self.cfg.t_release_min, self.cfg.t_release_max),
            (self.cfg.det_delay_min, self.cfg.det_delay_max),
            (self.cfg.u_speed_min, self.cfg.u_speed_max),
        ]
        return SearchSpace(names=names, bounds=bounds)

    def _evaluate_problem1(self, x: Sequence[float]) -> float:
        st = self.scenario.initial_states()
        t_release = float(x[0])
        det_delay = float(x[1])
        u_speed = float(x[2])
        bomb = BombEvent(t_release=t_release, t_detonate=t_release + det_delay)
        result = simulate_single_bomb(
            m0=st['m0'], u0=st['u0'], heading_to_xy=st['heading_to_xy'], u_speed=u_speed,
            bomb=bomb, true_target=st['true_target'], t_max=st['t_max']
        )
        return float(result.total_time)

    # ---------------- Problem 2 ----------------
    def _build_space_problem2(self) -> SearchSpace:
        names = ["theta", "u_speed", "t_release", "det_delay"]
        bounds = [
            (self.cfg.heading_min, self.cfg.heading_max),
            (self.cfg.u_speed_min, self.cfg.u_speed_max),
            (self.cfg.t_release_min, self.cfg.t_release_max),
            (self.cfg.det_delay_min, self.cfg.det_delay_max),
        ]
        return SearchSpace(names=names, bounds=bounds)

    def _evaluate_problem2(self, x: Sequence[float]) -> float:
        st = self.scenario.initial_states()
        theta = float(x[0])
        u_speed = float(x[1])
        t_release = float(x[2])
        det_delay = float(x[3])
        # Build heading target in XY from theta wrt UAV initial position
        u0 = st['u0']
        # Unit vector at angle theta
        import math
        dir_xy = (math.cos(theta), math.sin(theta))
        heading_to = (u0[0] + dir_xy[0], u0[1] + dir_xy[1], 0.0)
        bomb = BombEvent(t_release=t_release, t_detonate=t_release + det_delay)
        result = simulate_single_bomb(
            m0=st['m0'], u0=u0, heading_to_xy=heading_to, u_speed=u_speed,
            bomb=bomb, true_target=st['true_target'], t_max=st['t_max']
        )
        return float(result.total_time)

    def train(self):
        if isinstance(self.scenario, Problem1):
            space = self._build_space_problem1()
            evaluator = self._evaluate_problem1
        elif isinstance(self.scenario, Problem2):
            space = self._build_space_problem2()
            evaluator = self._evaluate_problem2
        else:
            # 暂用Problem2的参数空间作为占位
            space = self._build_space_problem2()
            evaluator = self._evaluate_problem2

        opt = self.OptimizerCls(seed=self.cfg.seed)

        def log_callback(step: int, val: float, x: Sequence[float]):
            if self.writer is not None:
                self.writer.add_scalar('objective/obscuration_time', val, step)
                for i, name in enumerate(space.names):
                    self.writer.add_scalar(f'params/{name}', float(x[i]), step)

        best_val, best_x = opt.optimize(evaluate_fn=evaluator, search_space=space, max_steps=self.cfg.steps, log_callback=log_callback)
        if self.writer is not None:
            for i, name in enumerate(space.names):
                self.writer.add_scalar(f'best/{name}', float(best_x[i]), 0)
            self.writer.add_scalar('best/obscuration_time', best_val, 0)
            self.writer.flush()
        return best_val, best_x


