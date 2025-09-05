# -*- coding: utf-8 -*-
"""Algorithms registry and convenience imports."""
from typing import Dict, Type

from .base import Optimizer
from .random_search import RandomSearch


REGISTRY: Dict[str, Type[Optimizer]] = {
    "random_search": RandomSearch,
}


def get_optimizer(name: str) -> Type[Optimizer]:
    key = name.lower().strip()
    if key not in REGISTRY:
        raise KeyError(f"Unknown optimizer: {name}")
    return REGISTRY[key]


