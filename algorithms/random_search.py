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


