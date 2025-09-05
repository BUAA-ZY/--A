# -*- coding: utf-8 -*-
"""Algorithms registry and convenience imports."""
from typing import Dict, Type

from .base import Optimizer
from .random_search import RandomSearch
from .hill_climb import HillClimb
from .simulated_annealing import SimulatedAnnealing
from .nelder_mead import NelderMead
from .pso import PSO
from .ga import GeneticAlgorithm


REGISTRY: Dict[str, Type[Optimizer]] = {
    "random_search": RandomSearch,
    "hill_climb": HillClimb,
    "sa": SimulatedAnnealing,
    "nelder_mead": NelderMead,
    "pso": PSO,
    "ga": GeneticAlgorithm,
}


def get_optimizer(name: str) -> Type[Optimizer]:
    key = name.lower().strip()
    if key not in REGISTRY:
        raise KeyError(f"Unknown optimizer: {name}")
    return REGISTRY[key]


