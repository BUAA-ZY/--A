# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, List, Sequence, Tuple
import numpy as np

from .base import Optimizer, SearchSpace


class RandomSearch(Optimizer):
    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self.rng = np.random.default_rng(self.seed)

    def _sample(self, space: SearchSpace) -> List[float]:
        xs: List[float] = []
        for (lo, hi) in space.bounds:
            xs.append(float(self.rng.uniform(lo, hi)))
        return xs

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

        best_val = float('-inf')
        best_x: List[float] = []
        for step in range(max_steps):
            x = self._sample(search_space)
            val = float(evaluate_fn(x))
            if val > best_val:
                best_val = val
                best_x = list(x)
            if log_callback is not None:
                log_callback(step, val, x)
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
        best_val = float('-inf')
        best_x: List[float] = []
        # 以批 32 进行评估
        batch_size = 32
        steps_done = 0
        while steps_done < max_steps:
            cur = min(batch_size, max_steps - steps_done)
            xs = [self._sample(search_space) for _ in range(cur)]
            vals = batch_evaluate_fn(xs)
            for i in range(cur):
                val = float(vals[i])
                x = xs[i]
                if val > best_val:
                    best_val = val
                    best_x = list(x)
                if log_callback is not None:
                    log_callback(steps_done + i, val, x)
            steps_done += cur
        return best_val, best_x


