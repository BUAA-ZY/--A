# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Callable, List, Sequence, Tuple
import numpy as np

from .base import Optimizer, SearchSpace


class GeneticAlgorithm(Optimizer):
    def __init__(self, seed: int = 42, pop_size: int = 40, elite_frac: float = 0.2, mut_rate: float = 0.2, mut_scale: float = 0.1):
        super().__init__(seed)
        self.rng = np.random.default_rng(self.seed)
        self.pop_size = int(pop_size)
        self.elite_frac = float(elite_frac)
        self.mut_rate = float(mut_rate)
        self.mut_scale = float(mut_scale)

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

        def init_pop(n: int) -> np.ndarray:
            return lo + self.rng.random((n, d)) * width

        def clip(pop: np.ndarray) -> np.ndarray:
            return np.minimum(np.maximum(pop, lo), hi)

        pop = init_pop(self.pop_size)
        scores = np.array([evaluate_fn(ind.tolist()) for ind in pop], dtype=float)
        elite_n = max(1, int(self.elite_frac * self.pop_size))

        for gen in range(1, max_steps + 1):
            order = np.argsort(scores)[::-1]
            pop = pop[order]
            scores = scores[order]
            elites = pop[:elite_n]

            # Crossover (uniform)
            children = []
            while len(children) < self.pop_size - elite_n:
                a, b = self.rng.integers(0, elite_n), self.rng.integers(0, elite_n)
                mask = self.rng.random(d) < 0.5
                child = np.where(mask, elites[a], elites[b])
                children.append(child)
            children = np.array(children, dtype=float)

            # Mutation
            if self.mut_rate > 0.0:
                mut_mask = self.rng.random(children.shape) < self.mut_rate
                noise = (self.rng.normal(0.0, 1.0, size=children.shape) * (self.mut_scale * width))
                children = np.where(mut_mask, children + noise, children)

            pop = np.vstack([elites, children])
            pop = clip(pop)
            scores = np.array([evaluate_fn(ind.tolist()) for ind in pop], dtype=float)
            if log_callback is not None:
                log_callback(gen, float(scores.max()), pop[int(np.argmax(scores))].tolist())

        idx = int(np.argmax(scores))
        return float(scores[idx]), pop[idx].tolist()

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

        rng = np.random.default_rng(self.seed)
        lo = np.array([b[0] for b in search_space.bounds], dtype=float)
        hi = np.array([b[1] for b in search_space.bounds], dtype=float)
        width = hi - lo

        def init_pop(n: int) -> np.ndarray:
            return lo + rng.random((n, d)) * width

        def clip(pop: np.ndarray) -> np.ndarray:
            return np.minimum(np.maximum(pop, lo), hi)

        pop = init_pop(self.pop_size)
        scores = np.array(batch_evaluate_fn([ind.tolist() for ind in pop]), dtype=float)
        elite_n = max(1, int(self.elite_frac * self.pop_size))

        for gen in range(1, max_steps + 1):
            order = np.argsort(scores)[::-1]
            pop = pop[order]
            scores = scores[order]
            elites = pop[:elite_n]
            children = []
            while len(children) < self.pop_size - elite_n:
                a, b = rng.integers(0, elite_n), rng.integers(0, elite_n)
                mask = rng.random(d) < 0.5
                child = np.where(mask, elites[a], elites[b])
                children.append(child)
            children = np.array(children, dtype=float)
            if self.mut_rate > 0.0:
                mut_mask = rng.random(children.shape) < self.mut_rate
                noise = (rng.normal(0.0, 1.0, size=children.shape) * (self.mut_scale * width))
                children = np.where(mut_mask, children + noise, children)
            pop = np.vstack([elites, children])
            pop = clip(pop)
            scores = np.array(batch_evaluate_fn([ind.tolist() for ind in pop]), dtype=float)
            if log_callback is not None:
                log_callback(gen, float(scores.max()), pop[int(np.argmax(scores))].tolist())
        idx = int(np.argmax(scores))
        return float(scores[idx]), pop[idx].tolist()


