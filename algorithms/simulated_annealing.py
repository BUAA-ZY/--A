# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, List, Sequence, Tuple
import numpy as np

from .base import Optimizer, SearchSpace


class SimulatedAnnealing(Optimizer):
    def __init__(self, seed: int = 42, t0: float = 1.0, t_min: float = 1e-3, alpha: float = 0.95):
        super().__init__(seed)
        self.rng = np.random.default_rng(self.seed)
        self.t0 = float(t0)
        self.t_min = float(t_min)
        self.alpha = float(alpha)

    def _rand_in_bounds(self, bounds: Sequence[Tuple[float, float]]) -> List[float]:
        return [float(self.rng.uniform(lo, hi)) for (lo, hi) in bounds]

    def _clip(self, x: Sequence[float], bounds: Sequence[Tuple[float, float]]) -> List[float]:
        y: List[float] = []
        for i, (lo, hi) in enumerate(bounds):
            y.append(float(min(hi, max(lo, x[i]))))
        return y

    def optimize(
        self,
        evaluate_fn: Callable[[Sequence[float]], float],
        search_space: SearchSpace,
        max_steps: int,
        log_callback: Callable[[int, float, Sequence[float]], None] | None = None,
    ) -> Tuple[float, List[float]]:
        if search_space.dim == 0:
            val = float(evaluate_fn([]))
            if log_callback is not None:
                log_callback(0, val, [])
            return val, []

        widths = np.array([b[1] - b[0] for b in search_space.bounds], dtype=float)
        sigma0 = 0.2 * np.maximum(widths, 1e-9)
        x = self._rand_in_bounds(search_space.bounds)
        fx = float(evaluate_fn(x))
        best_x = list(x)
        best_val = fx
        t = self.t0
        if log_callback is not None:
            log_callback(0, fx, x)

        step = 1
        while step <= max_steps and t > self.t_min:
            noise = self.rng.normal(0.0, 1.0, size=search_space.dim) * sigma0 * (t / self.t0)
            cand = [float(x[i] + noise[i]) for i in range(search_space.dim)]
            cand = self._clip(cand, search_space.bounds)
            fc = float(evaluate_fn(cand))
            if fc > fx or self.rng.uniform() < np.exp((fc - fx) / max(1e-9, t)):
                x, fx = cand, fc
                if fx > best_val:
                    best_val = fx
                    best_x = list(x)
            if log_callback is not None:
                log_callback(step, fc, cand)
            t *= self.alpha
            step += 1
        return best_val, best_x

    def optimize_batch(
        self,
        evaluate_fn: Callable[[Sequence[float]], float],
        search_space: SearchSpace,
        max_steps: int,
        batch_evaluate_fn: Callable[[Sequence[Sequence[float]]], List[float]] | None = None,
        log_callback: Callable[[int, float, Sequence[float]], None] | None = None,
    ) -> Tuple[float, List[float]]:
        if batch_evaluate_fn is None:
            return self.optimize(evaluate_fn, search_space, max_steps, log_callback)

        widths = np.array([b[1] - b[0] for b in search_space.bounds], dtype=float)
        sigma0 = 0.2 * np.maximum(widths, 1e-9)
        x = [float(np.random.default_rng(self.seed).uniform(lo, hi)) for (lo, hi) in search_space.bounds]
        fx = float(batch_evaluate_fn([x])[0])
        best_x = list(x)
        best_val = fx
        t = self.t0
        if log_callback is not None:
            log_callback(0, fx, x)

        step = 1
        while step <= max_steps and t > self.t_min:
            # 以小批量候选提高吞吐
            cur = min(32, max_steps - step + 1)
            noise = np.random.default_rng(self.seed + step).normal(0.0, 1.0, size=(cur, search_space.dim)) * sigma0 * (t / self.t0)
            xs = []
            for i in range(cur):
                cand = [float(x[j] + noise[i, j]) for j in range(search_space.dim)]
                # clip
                cand = [float(min(search_space.bounds[j][1], max(search_space.bounds[j][0], cand[j]))) for j in range(search_space.dim)]
                xs.append(cand)
            vals = batch_evaluate_fn(xs)
            # 采用第一条候选的Metropolis判定来更新当前解（简化版）
            fc = float(vals[0])
            cand0 = xs[0]
            if fc > fx or np.random.default_rng(self.seed + step).uniform() < np.exp((fc - fx) / max(1e-9, t)):
                x, fx = cand0, fc
                if fx > best_val:
                    best_val, best_x = fx, list(x)
            if log_callback is not None:
                log_callback(step, fc, cand0)
            t *= self.alpha
            step += 1
        return best_val, best_x


