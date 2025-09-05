# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, List, Sequence, Tuple
import numpy as np

from .base import Optimizer, SearchSpace


class NelderMead(Optimizer):
    def __init__(self, seed: int = 42, alpha: float = 1.0, gamma: float = 2.0, rho: float = 0.5, sigma: float = 0.5):
        super().__init__(seed)
        self.rng = np.random.default_rng(self.seed)
        self.alpha = float(alpha)  # reflection
        self.gamma = float(gamma)  # expansion
        self.rho = float(rho)      # contraction
        self.sigma = float(sigma)  # shrink

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
        n = search_space.dim
        if n == 0:
            val = float(evaluate_fn([]))
            if log_callback is not None:
                log_callback(0, val, [])
            return val, []

        # Initialize simplex
        x0 = np.array(self._rand_in_bounds(search_space.bounds), dtype=float)
        widths = np.array([b[1] - b[0] for b in search_space.bounds], dtype=float)
        simplex = [x0]
        for i in range(n):
            xi = np.array(x0, copy=True)
            xi[i] = float(min(search_space.bounds[i][1], max(search_space.bounds[i][0], xi[i] + 0.05 * max(1e-9, widths[i]))))
            simplex.append(xi)

        fvals = [float(evaluate_fn(p.tolist())) for p in simplex]
        step = 0
        while step < max_steps:
            # Order
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]
            x_best, f_best = simplex[-1], fvals[-1]
            x_worst, f_worst = simplex[0], fvals[0]
            x_second_worst = simplex[-2]

            if log_callback is not None:
                log_callback(step, f_best, x_best.tolist())

            # Centroid of all but worst
            centroid = np.mean(simplex[1:], axis=0)
            # Reflection
            xr = centroid + self.alpha * (centroid - x_worst)
            xr = np.array(self._clip(xr.tolist(), search_space.bounds))
            fr = float(evaluate_fn(xr.tolist()))

            if fr > fvals[-1]:
                # Expansion
                xe = centroid + self.gamma * (xr - centroid)
                xe = np.array(self._clip(xe.tolist(), search_space.bounds))
                fe = float(evaluate_fn(xe.tolist()))
                if fe > fr:
                    simplex[0] = xe
                    fvals[0] = fe
                else:
                    simplex[0] = xr
                    fvals[0] = fr
            elif fr > fvals[-2]:
                simplex[0] = xr
                fvals[0] = fr
            else:
                # Contraction
                xc = centroid + self.rho * (x_worst - centroid)
                xc = np.array(self._clip(xc.tolist(), search_space.bounds))
                fc = float(evaluate_fn(xc.tolist()))
                if fc > fvals[0]:
                    simplex[0] = xc
                    fvals[0] = fc
                else:
                    # Shrink towards best
                    xb = simplex[-1]
                    new_simplex = [xb]
                    new_fvals = [fvals[-1]]
                    for i in range(len(simplex) - 1):
                        xs = xb + self.sigma * (simplex[i] - xb)
                        xs = np.array(self._clip(xs.tolist(), search_space.bounds))
                        new_simplex.append(xs)
                        new_fvals.append(float(evaluate_fn(xs.tolist())))
                    simplex, fvals = new_simplex, new_fvals
            step += 1

        # Return best
        idx = np.argmax(fvals)
        return float(fvals[idx]), simplex[idx].tolist()

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
        n = search_space.dim
        if n == 0:
            val = float(evaluate_fn([]))
            if log_callback is not None:
                log_callback(0, val, [])
            return val, []

        rng = np.random.default_rng(int(self.seed))
        x0 = np.array([float(rng.uniform(lo, hi)) for (lo, hi) in search_space.bounds], dtype=float)
        widths = np.array([b[1] - b[0] for b in search_space.bounds], dtype=float)
        simplex = [x0]
        for i in range(n):
            xi = np.array(x0, copy=True)
            xi[i] = float(min(search_space.bounds[i][1], max(search_space.bounds[i][0], xi[i] + 0.05 * max(1e-9, widths[i]))))
            simplex.append(xi)

        fvals = batch_evaluate_fn([p.tolist() for p in simplex])
        step = 0
        while step < max_steps:
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]
            x_best, f_best = simplex[-1], fvals[-1]
            x_worst, f_worst = simplex[0], fvals[0]
            if log_callback is not None:
                log_callback(step, float(f_best), simplex[-1].tolist())

            centroid = np.mean(simplex[1:], axis=0)
            xr = centroid + self.alpha * (centroid - x_worst)
            xr = np.clip(xr, [b[0] for b in search_space.bounds], [b[1] for b in search_space.bounds])
            # 批评估：反射、扩展、收缩同时打包
            xe = centroid + self.gamma * (xr - centroid)
            xc = centroid + self.rho * (x_worst - centroid)
            xe = np.clip(xe, [b[0] for b in search_space.bounds], [b[1] for b in search_space.bounds])
            xc = np.clip(xc, [b[0] for b in search_space.bounds], [b[1] for b in search_space.bounds])
            vals = batch_evaluate_fn([xr.tolist(), xe.tolist(), xc.tolist()])
            fr, fe, fc = float(vals[0]), float(vals[1]), float(vals[2])

            if fr > fvals[-1]:
                if fe > fr:
                    simplex[0], fvals[0] = xe, fe
                else:
                    simplex[0], fvals[0] = xr, fr
            elif fr > fvals[-2]:
                simplex[0], fvals[0] = xr, fr
            else:
                if fc > fvals[0]:
                    simplex[0], fvals[0] = xc, fc
                else:
                    # shrink
                    xb = simplex[-1]
                    candidates = []
                    new_simplex = [xb]
                    new_fvals = [fvals[-1]]
                    for i in range(len(simplex) - 1):
                        xs = xb + self.sigma * (simplex[i] - xb)
                        xs = np.clip(xs, [b[0] for b in search_space.bounds], [b[1] for b in search_space.bounds])
                        new_simplex.append(xs)
                        candidates.append(xs.tolist())
                    cand_vals = batch_evaluate_fn(candidates)
                    new_fvals.extend([float(v) for v in cand_vals])
                    simplex, fvals = new_simplex, new_fvals
            step += 1
        idx = int(np.argmax(fvals))
        return float(fvals[idx]), simplex[idx].tolist()


