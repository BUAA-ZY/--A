# -*- coding: utf-8 -*-
"""Geometry helpers: vectors, distances, projections, and intersection tests."""
from typing import Optional, Tuple
import math

from .constants import EPS
from .types import Vec3


def norm(v: Vec3) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def unit(v: Vec3) -> Vec3:
    n = norm(v)
    if n <= EPS:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def distance_point_to_segment(p: Vec3, a: Vec3, b: Vec3) -> float:
    ab = sub(b, a)
    ap = sub(p, a)
    ab2 = dot(ab, ab)
    if ab2 <= EPS:
        return norm(ap)
    t = max(0.0, min(1.0, dot(ap, ab) / ab2))
    proj = add(a, scale(ab, t))
    return norm(sub(p, proj))


def line_of_sight_point(tgt: Vec3, src: Vec3, alpha: float) -> Vec3:
    """Point on segment src->tgt at fraction alpha in [0,1]."""
    return (src[0] + (tgt[0] - src[0]) * alpha,
            src[1] + (tgt[1] - src[1]) * alpha,
            src[2] + (tgt[2] - src[2]) * alpha)


def solve_quadratic(a: float, b: float, c: float) -> Optional[Tuple[float, float]]:
    if abs(a) <= EPS:
        if abs(b) <= EPS:
            return None
        x = -c / b
        return (x, x)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    s = math.sqrt(max(0.0, disc))
    return ((-b - s) / (2.0 * a), (-b + s) / (2.0 * a))


