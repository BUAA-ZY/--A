# -*- coding: utf-8 -*-
"""Occlusion (obscuration) tests between missile and true target by a spherical cloud."""
from typing import List, Optional, Tuple

from .constants import CLOUD_RADIUS
from .geometry import distance_point_to_segment
from .types import CylinderTarget, Vec3


def is_obscured_by_sphere(missile_pos: Vec3, target_point: Vec3, cloud_center: Vec3, cloud_radius: float = CLOUD_RADIUS) -> bool:
    """Return True if the segment missile_pos->target_point intersects (within radius) the sphere."""
    d = distance_point_to_segment(cloud_center, missile_pos, target_point)
    return d <= cloud_radius


def sample_target_points(target: CylinderTarget, num_side: int = 12, include_top: bool = True) -> List[Vec3]:
    """Sample target points on cylinder surface and optionally top center.

    This allows conservative occlusion via union over samples.
    """
    cx, cy, cz = target.center
    pts: List[Vec3] = []
    for i in range(num_side):
        ang = 2.0 * 3.141592653589793 * i / num_side
        pts.append((cx + target.radius * __import__('math').cos(ang),
                    cy + target.radius * __import__('math').sin(ang),
                    cz + 0.5 * target.height))
    if include_top:
        pts.append((cx, cy, cz + target.height))
    # Include center of side mid-height as approximate
    pts.append((cx, cy, cz + 0.5 * target.height))
    return pts


