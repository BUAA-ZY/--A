# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, List, Sequence, Tuple
import numpy as np

from .base import Optimizer, SearchSpace


class PSO(Optimizer):
    def __init__(self, seed: int = 42, swarm_size: int = 24, w: float = 0.7, c1: float = 1.5, c2: float = 1.5):
        super().__init__(seed)
        self.rng = np.random.default_rng(self.seed)
        self.swarm_size = int(swarm_size)
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)

    def optimize(
        self,
        evaluate_fn: Callable[[Sequence[float]], float],
        search_space: SearchSpace,
        max_steps: int,
        log_callback: Callable[[int, float, Sequence[float]], None] | None = None,
    ) -> Tuple[float, List[float]]:
        d = search_space.dim
        if d == 0:
            val = float(evaluate_fn([]))
            if log_callback is not None:
                log_callback(0, val, [])
            return val, []
        lo = np.array([b[0] for b in search_space.bounds], dtype=float)
        hi = np.array([b[1] for b in search_space.bounds], dtype=float)
        width = hi - lo
        pos = lo + self.rng.random((self.swarm_size, d)) * width
        vel = 0.1 * (self.rng.random((self.swarm_size, d)) - 0.5) * width
        pbest = pos.copy()
        pbest_val = np.array([evaluate_fn(p.tolist()) for p in pos], dtype=float)
        g_idx = int(np.argmax(pbest_val))
        gbest = pbest[g_idx].copy()
        gbest_val = float(pbest_val[g_idx])

        for step in range(1, max_steps + 1):
            r1 = self.rng.random((self.swarm_size, d))
            r2 = self.rng.random((self.swarm_size, d))
            vel = self.w * vel + self.c1 * r1 * (pbest - pos) + self.c2 * r2 * (gbest - pos)
            pos = pos + vel
            pos = np.minimum(np.maximum(pos, lo), hi)
            vals = np.array([evaluate_fn(p.tolist()) for p in pos], dtype=float)
            improved = vals > pbest_val
            pbest[improved] = pos[improved]
            pbest_val[improved] = vals[improved]
            if float(vals.max()) > gbest_val:
                g_idx = int(np.argmax(vals))
                gbest = pos[g_idx].copy()
                gbest_val = float(vals[g_idx])
            if log_callback is not None:
                log_callback(step, gbest_val, gbest.tolist())
        return gbest_val, gbest.tolist()

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
        d = search_space.dim
        if d == 0:
            val = float(evaluate_fn([]))
            if log_callback is not None:
                log_callback(0, val, [])
            return val, []
        lo = np.array([b[0] for b in search_space.bounds], dtype=float)
        hi = np.array([b[1] for b in search_space.bounds], dtype=float)
        width = hi - lo
        pos = lo + np.random.default_rng(self.seed).random((self.swarm_size, d)) * width
        vel = 0.1 * (np.random.default_rng(self.seed + 1).random((self.swarm_size, d)) - 0.5) * width
        # 批量评估
        pbest = pos.copy()
        pbest_val = np.array(batch_evaluate_fn([p.tolist() for p in pos]), dtype=float)
        g_idx = int(np.argmax(pbest_val))
        gbest = pbest[g_idx].copy()
        gbest_val = float(pbest_val[g_idx])

        for step in range(1, max_steps + 1):
            r1 = np.random.default_rng(self.seed + step).random((self.swarm_size, d))
            r2 = np.random.default_rng(self.seed + step + 1).random((self.swarm_size, d))
            vel = self.w * vel + self.c1 * r1 * (pbest - pos) + self.c2 * r2 * (gbest - pos)
            pos = pos + vel
            pos = np.minimum(np.maximum(pos, lo), hi)
            vals = np.array(batch_evaluate_fn([p.tolist() for p in pos]), dtype=float)
            improved = vals > pbest_val
            pbest[improved] = pos[improved]
            pbest_val[improved] = vals[improved]
            if float(vals.max()) > gbest_val:
                g_idx = int(np.argmax(vals))
                gbest = pos[g_idx].copy()
                gbest_val = float(vals[g_idx])
            if log_callback is not None:
                log_callback(step, gbest_val, gbest.tolist())
        return gbest_val, gbest.tolist()


