# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, List, Sequence, Tuple
import numpy as np

from .base import Optimizer, SearchSpace


class HillClimb(Optimizer):
    def __init__(self, seed: int = 42, step_scale: float = 0.1):
        super().__init__(seed)
        self.rng = np.random.default_rng(self.seed)
        self.step_scale = float(step_scale)

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

        # Initialize
        widths = np.array([b[1] - b[0] for b in search_space.bounds], dtype=float)
        sigma = self.step_scale * np.maximum(widths, 1e-9)
        x = self._rand_in_bounds(search_space.bounds)
        best_val = float(evaluate_fn(x))
        best_x = list(x)
        if log_callback is not None:
            log_callback(0, best_val, best_x)

        for step in range(1, max_steps + 1):
            noise = self.rng.normal(0.0, 1.0, size=search_space.dim) * sigma
            cand = [float(best_x[i] + noise[i]) for i in range(search_space.dim)]
            cand = self._clip(cand, search_space.bounds)
            val = float(evaluate_fn(cand))
            if val > best_val:
                best_val = val
                best_x = cand
            if log_callback is not None:
                log_callback(step, val, cand)
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
        sigma = self.step_scale * np.maximum(widths, 1e-9)
        x = self._rand_in_bounds(search_space.bounds)
        fx = float(batch_evaluate_fn([x])[0])
        best_x = list(x)
        best_val = fx
        if log_callback is not None:
            log_callback(0, fx, x)

        # 批量生成邻域候选，加速评估
        batch_size = 32
        step = 1
        while step <= max_steps:
            cur = min(batch_size, max_steps - step + 1)
            noise = self.rng.normal(0.0, 1.0, size=(cur, search_space.dim)) * sigma
            xs = []
            for i in range(cur):
                cand = [float(best_x[j] + noise[i, j]) for j in range(search_space.dim)]
                cand = self._clip(cand, search_space.bounds)
                xs.append(cand)
            vals = batch_evaluate_fn(xs)
            for i in range(cur):
                val, cand = float(vals[i]), xs[i]
                if val > best_val:
                    best_val, best_x = val, cand
                if log_callback is not None:
                    log_callback(step, val, cand)
                step += 1
                if step > max_steps:
                    break
        return best_val, best_x


