"""Physics module: constants, geometry, dynamics, occlusion, and simulation.

This package provides the core physical models and utility functions used by
the optimization/training code and by scenario definitions.
"""

from . import constants as constants
from . import geometry as geometry
from . import dynamics as dynamics
from . import occlusion as occlusion
from . import simulation as simulation
from . import types as types

__all__ = [
    "constants",
    "geometry",
    "dynamics",
    "occlusion",
    "simulation",
    "types",
]


