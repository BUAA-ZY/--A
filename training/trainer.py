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
from physics.simulation import simulate_single_bomb, simulate_multi_bombs
from physics.types import BombEvent, CylinderTarget
from .config import TrainConfig
def _evaluate_worker(args):
    """Top-level worker for multiprocessing evaluation.

    args = (scenario_key: str, x: Sequence[float])
    Returns float objective value.
    """
    scenario_key, x = args
    # Local imports inside worker for safety
    from scenarios import Problem1, Problem2, Problem3
    from physics.simulation import simulate_single_bomb, simulate_multi_bombs
    from physics.types import BombEvent, CylinderTarget
    import math

    # Build scenario object and states
    if scenario_key == 'problem1':
        sc = Problem1()
        st = sc.initial_states()
        t_release = float(x[0])
        det_delay = float(x[1])
        u_speed = float(x[2])
        bomb = BombEvent(t_release=t_release, t_detonate=t_release + det_delay)
        res = simulate_single_bomb(m0=st['m0'], u0=st['u0'], heading_to_xy=st['heading_to_xy'], u_speed=u_speed, bomb=bomb, true_target=st['true_target'], t_max=st['t_max'])
        return float(res.total_time)
    elif scenario_key == 'problem2':
        sc = Problem2()
        st = sc.initial_states()
        theta = float(x[0])
        u_speed = float(x[1])
        t_release = float(x[2])
        det_delay = float(x[3])
        u0 = st['u0']
        dir_xy = (math.cos(theta), math.sin(theta))
        heading_to = (u0[0] + dir_xy[0], u0[1] + dir_xy[1], 0.0)
        bomb = BombEvent(t_release=t_release, t_detonate=t_release + det_delay)
        res = simulate_single_bomb(m0=st['m0'], u0=u0, heading_to_xy=heading_to, u_speed=u_speed, bomb=bomb, true_target=st['true_target'], t_max=st['t_max'])
        return float(res.total_time)
    elif scenario_key == 'problem3':
        sc = Problem3()
        st = sc.initial_states()
        theta = float(x[0])
        u_speed = float(x[1])
        u0 = st['u0']
        dir_xy = (math.cos(theta), math.sin(theta))
        heading_to = (u0[0] + dir_xy[0], u0[1] + dir_xy[1], 0.0)
        bombs = [
            BombEvent(t_release=float(x[2]), t_detonate=float(x[2]) + float(x[3])),
            BombEvent(t_release=float(x[4]), t_detonate=float(x[4]) + float(x[5])),
            BombEvent(t_release=float(x[6]), t_detonate=float(x[6]) + float(x[7])),
        ]
        tr = sorted([b.t_release for b in bombs])
        ok = all(tr[i+1] - tr[i] >= 1.0 for i in range(len(tr) - 1))
        if not ok:
            return -1e6
        res = simulate_multi_bombs(m0=st['m0'], u0=u0, heading_to_xy=heading_to, u_speed=u_speed, bombs=bombs, true_target=st['true_target'], t_max=st['t_max'])
        return float(res.total_time)
    else:
        # Fallback use Problem2 parameterization
        sc = Problem2()
        st = sc.initial_states()
        theta = float(x[0])
        u_speed = float(x[1])
        t_release = float(x[2])
        det_delay = float(x[3])
        u0 = st['u0']
        dir_xy = (math.cos(theta), math.sin(theta))
        heading_to = (u0[0] + dir_xy[0], u0[1] + dir_xy[1], 0.0)
        bomb = BombEvent(t_release=t_release, t_detonate=t_release + det_delay)
        res = simulate_single_bomb(m0=st['m0'], u0=u0, heading_to_xy=heading_to, u_speed=u_speed, bomb=bomb, true_target=st['true_target'], t_max=st['t_max'])
        return float(res.total_time)


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
        # 并行批评估调试计数
        self._batch_debug_count = 0
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

    # ---------------- Problem 3 ----------------
    def _build_space_problem3(self) -> SearchSpace:
        # x = [theta, u_speed, t_r1, d1, t_r2, d2, t_r3, d3]
        names = [
            "theta", "u_speed",
            "t_release_1", "det_delay_1",
            "t_release_2", "det_delay_2",
            "t_release_3", "det_delay_3",
        ]
        b = [
            (self.cfg.heading_min, self.cfg.heading_max),
            (self.cfg.u_speed_min, self.cfg.u_speed_max),
            (self.cfg.t_release_min, self.cfg.t_release_max),
            (self.cfg.det_delay_min, self.cfg.det_delay_max),
            (self.cfg.t_release_min, self.cfg.t_release_max),
            (self.cfg.det_delay_min, self.cfg.det_delay_max),
            (self.cfg.t_release_min, self.cfg.t_release_max),
            (self.cfg.det_delay_min, self.cfg.det_delay_max),
        ]
        return SearchSpace(names=names, bounds=b)

    def _evaluate_problem3(self, x: Sequence[float]) -> float:
        st = self.scenario.initial_states()
        import math
        theta = float(x[0])
        u_speed = float(x[1])
        u0 = st['u0']
        dir_xy = (math.cos(theta), math.sin(theta))
        heading_to = (u0[0] + dir_xy[0], u0[1] + dir_xy[1], 0.0)
        bombs = [
            BombEvent(t_release=float(x[2]), t_detonate=float(x[2]) + float(x[3])),
            BombEvent(t_release=float(x[4]), t_detonate=float(x[4]) + float(x[5])),
            BombEvent(t_release=float(x[6]), t_detonate=float(x[6]) + float(x[7])),
        ]
        # Enforce min 1s interval (hard penalty)
        tr = sorted([b.t_release for b in bombs])
        ok = all(tr[i+1] - tr[i] >= 1.0 for i in range(len(tr) - 1))
        if not ok:
            return -1e6
        res = simulate_multi_bombs(
            m0=st['m0'], u0=u0, heading_to_xy=heading_to, u_speed=u_speed,
            bombs=bombs, true_target=st['true_target'], t_max=st['t_max']
        )
        # Real-time print (trainer will call many times; keep concise)
        # print(f"theta={theta:.3f}, u={u_speed:.1f}, total={res.total_time:.3f}")
        if self.writer is not None:
            # Detailed TB logs
            import time as _time
            step = int(_time.time())  # coarse wall-clock as global step
            self.writer.add_scalar('q3/total_time', float(res.total_time), step)
            for i, t in enumerate(res.per_bomb_times):
                self.writer.add_scalar(f'q3/per_bomb_time_{i+1}', float(t), step)
            self.writer.flush()
        return float(res.total_time)

    def train(self):
        if isinstance(self.scenario, Problem1):
            space = self._build_space_problem1()
            evaluator = self._evaluate_problem1
        elif isinstance(self.scenario, Problem2):
            space = self._build_space_problem2()
            evaluator = self._evaluate_problem2
        elif isinstance(self.scenario, Problem3):
            space = self._build_space_problem3()
            evaluator = self._evaluate_problem3
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

        # 并行批评估包装
        def batch_evaluate_fn(xs):
            if self.cfg.max_workers and self.cfg.max_workers > 1:
                from concurrent.futures import ProcessPoolExecutor
                # Debug print: first few calls only
                self._batch_debug_count += 1
                if self._batch_debug_count <= 3:
                    print(f"[Trainer] Parallel batch evaluate: batch_size={len(xs)}, max_workers={self.cfg.max_workers}")
                scenario_key = self.cfg.scenario.lower().strip()
                payloads = [(scenario_key, list(x)) for x in xs]
                with ProcessPoolExecutor(max_workers=self.cfg.max_workers) as ex:
                    vals = list(ex.map(_evaluate_worker, payloads))
                return [float(v) for v in vals]
            else:
                return [float(evaluator(x)) for x in xs]

        # 优先调用支持批评估的接口
        try:
            best_val, best_x = opt.optimize_batch(
                evaluate_fn=evaluator, search_space=space, max_steps=self.cfg.steps,
                batch_evaluate_fn=batch_evaluate_fn, log_callback=log_callback,
            )
        except Exception:
            best_val, best_x = opt.optimize(evaluate_fn=evaluator, search_space=space, max_steps=self.cfg.steps, log_callback=log_callback)
        if self.writer is not None:
            for i, name in enumerate(space.names):
                self.writer.add_scalar(f'best/{name}', float(best_x[i]), 0)
            self.writer.add_scalar('best/obscuration_time', best_val, 0)
            self.writer.flush()
        return best_val, best_x


