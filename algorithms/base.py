# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple
import abc


@dataclass
class SearchSpace:
    names: Sequence[str]
    bounds: Sequence[Tuple[float, float]]  # inclusive bounds

    def __post_init__(self):
        if len(self.names) != len(self.bounds):
            raise ValueError("names and bounds must have same length")

    @property
    def dim(self) -> int:
        return len(self.names)


class Optimizer(abc.ABC):
    def __init__(self, seed: int = 42):
        self.seed = int(seed)

    @abc.abstractmethod
    def optimize(
        self,
        evaluate_fn: Callable[[Sequence[float]], float],
        search_space: SearchSpace,
        max_steps: int,
        log_callback: Callable[[int, float, Sequence[float]], None] | None = None,
    ) -> Tuple[float, List[float]]:
        """Run optimization.

        Returns (best_value, best_params).
        """
        raise NotImplementedError

    # 可选：批量评估接口（默认逐一调用 evaluate_fn）
    def optimize_batch(
        self,
        evaluate_fn: Callable[[Sequence[float]], float],
        search_space: SearchSpace,
        max_steps: int,
        batch_evaluate_fn: Callable[[Sequence[Sequence[float]]], List[float]] | None = None,
        log_callback: Callable[[int, float, Sequence[float]], None] | None = None,
    ) -> Tuple[float, List[float]]:
        return self.optimize(evaluate_fn, search_space, max_steps, log_callback)


