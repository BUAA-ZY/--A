# -*- coding: utf-8 -*-
"""Matplotlib-based 3D renderer with axes and grid.

This renderer is independent from pygame and draws a 3D scene using
matplotlib's 3D Axes. It is suitable for plotting trajectories and agent
poses (x, y, z, heading) with a world coordinate frame.
"""
from typing import Dict, List, Tuple, Optional
import math
import time
import matplotlib
matplotlib.use('TkAgg')  # Fallback to TkAgg for interactive plotting on most systems
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed to enable 3D)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
import matplotlib.path as mpath
import numpy as np
import matplotlib.patches as mpatches


class Renderer3D:
    def __init__(self, title: str = 'UAV 3D 可视化', viewsize: Tuple[int, int, int] = (1000, 800, 600), trail_length: int = 200, trail_fade_seconds: float = 0.5, pause_sec: float = 0.001):
        self.title = title
        self.viewsize = viewsize
        self.fig = None
        self.ax = None
        self.trail_length = trail_length
        self.trail_fade_seconds = trail_fade_seconds
        self.pause_sec = pause_sec
        # Internal trails storage: id -> list of (x,y,z)
        self._trails: Dict[str, List[Tuple[float, float, float, float]]] = {}
        self._plane_marker = self._create_airplane_marker()
        # World bounds (can include negatives). Defaults to [0, viewsize]
        self.xlim: Tuple[float, float] = (0.0, float(viewsize[0]))
        self.ylim: Tuple[float, float] = (0.0, float(viewsize[1]))
        self.zlim: Tuple[float, float] = (0.0, float(viewsize[2]))
        # UI elements
        self._legend = None
        self._info_text = None
        # Interaction
        self._cid_scroll = None
        self._cid_key = None

    def init(self):
        self.fig = plt.figure(self.title, figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_zlim(*self.zlim)
        self.ax.grid(True, linestyle=':')
        self.ax.set_box_aspect((1, 1, 0.6))

    def clear(self):
        self.ax.cla()
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)
        self.ax.set_zlim(*self.zlim)
        self.ax.grid(True, linestyle=':')

    def set_bounds(self, xlim: Tuple[float, float], ylim: Tuple[float, float], zlim: Tuple[float, float]):
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        if self.ax is not None:
            self.ax.set_xlim(*self.xlim)
            self.ax.set_ylim(*self.ylim)
            self.ax.set_zlim(*self.zlim)

    # -----------------------------
    # Interaction (zoom)
    # -----------------------------
    def enable_interaction(self):
        if self.fig is None:
            return
        if self._cid_scroll is None:
            self._cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        if self._cid_key is None:
            self._cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_scroll(self, event):
        # Zoom in/out with mouse wheel; factor per tick
        factor = 0.9 if event.button == 'up' else 1.1
        self._zoom(factor)

    def _on_key(self, event):
        if event.key in ['+', '=']:
            self._zoom(0.9)
        elif event.key in ['-', '_']:
            self._zoom(1.1)
        elif event.key in ['r', 'R']:
            # Reset to stored bounds
            self.set_bounds(self.xlim, self.ylim, self.zlim)
            self.fig.canvas.draw_idle()

    def _zoom(self, factor: float):
        def scale_limits(lim: Tuple[float, float]) -> Tuple[float, float]:
            c = 0.5 * (lim[0] + lim[1])
            half = 0.5 * (lim[1] - lim[0]) * factor
            return (c - half, c + half)
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        cur_zlim = self.ax.get_zlim()
        self.ax.set_xlim(*scale_limits(cur_xlim))
        self.ax.set_ylim(*scale_limits(cur_ylim))
        self.ax.set_zlim(*scale_limits(cur_zlim))
        self.fig.canvas.draw_idle()

    def _create_airplane_marker(self):
        # Simple airplane-like 2D path (billboarded marker)
        Path = mpath.Path
        verts = [
            (1.2, 0.0),     # nose
            (0.0, 0.5),     # upper leading edge
            (-0.6, 0.2),    # wing upper
            (-1.2, 0.2),    # tail upper
            (-1.2, -0.2),   # tail lower
            (-0.6, -0.2),   # wing lower
            (0.0, -0.5),    # lower leading edge
            (1.2, 0.0),     # back to nose
        ]
        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
        ]
        # CLOSEPOLY requires one extra vertex; repeat the first
        verts.append(verts[0])
        codes.append(Path.CLOSEPOLY)
        return Path(verts, codes)

    def set_refresh_rate_hz(self, hz: float):
        if hz <= 0:
            self.pause_sec = 0.0
        else:
            self.pause_sec = max(0.0, 1.0 / hz)

    def _draw_aircraft(self, x: float, y: float, z: float, theta: float, color: str, size: float):
        # Simple 2D airplane polygon aligned by theta (radians) in XY plane
        # Base shape similar to _create_airplane_marker
        base = [
            (1.2, 0.0), (0.0, 0.5), (-0.6, 0.2), (-1.2, 0.2),
            (-1.2, -0.2), (-0.6, -0.2), (0.0, -0.5)
        ]
        s = max(1.0, size * 0.03)
        ct, st = math.cos(theta), math.sin(theta)
        verts2d = [
            (x + s * (ct * px - st * py), y + s * (st * px + ct * py))
            for (px, py) in base
        ]
        # Close polygon
        verts2d.append(verts2d[0])
        poly3d = [[(vx, vy, z) for (vx, vy) in verts2d]]
        self.ax.add_collection3d(Poly3DCollection(poly3d, facecolors=color, edgecolors=color, linewidths=0.5, alpha=1.0))

    def draw_scene(self, data: Dict):
        """Draw 3D scene.

        New unified API:
          - entities: List[Dict{id,x,y,z,theta?,color?,size?}]
          - obstacles: Optional[List[Dict{x,y,z,radius}]]
          - spheres: Optional[List[Dict{x,y,z,radius,color,alpha}]]
          - cylinders: Optional[List[Dict{x,y,z,height,radius,color,alpha}]] (z is base)
          - axes_origin: Optional[Tuple[x,y,z]] (default: (0,0,0))
        Automatically maintains trails per entity id.
        """
        self.clear()

        # Obstacles (optional)
        for obs in data.get('obstacles', []) or []:
            r = obs.get('radius', 20)
            phi = [i * math.pi / 50 for i in range(101)]
            cx = [obs['x'] + r * math.cos(p) for p in phi]
            cy = [obs['y'] + r * math.sin(p) for p in phi]
            cz = [obs['z'] for _ in phi]
            self.ax.plot(cx, cy, cz, color='k', linewidth=1)

        # Entities with airplane-shaped marker (use triangle marker as approximation)
        for ent in data.get('entities', []) or []:
            ent_id = str(ent.get('id', id(ent)))
            x, y, z = ent.get('x', 0), ent.get('y', 0), ent.get('z', 0)
            color = ent.get('color', 'g')
            size = ent.get('size', 50)
            # Determine heading: prefer provided theta, else compute from recent trail
            theta = ent.get('theta', None)
            trail_prev = self._trails.get(ent_id, [])
            if theta is None and len(trail_prev) >= 1:
                x_prev, y_prev, _, _ = trail_prev[-1]
                dx, dy = x - x_prev, y - y_prev
                if abs(dx) + abs(dy) > 1e-6:
                    theta = math.atan2(dy, dx)
            if theta is None:
                theta = 0.0
            # Draw airplane polygon oriented along theta
            self._draw_aircraft(x, y, z, theta, color, size)
            # Update and draw trail
            trail = self._trails.setdefault(ent_id, [])
            now = time.monotonic()
            trail.append((x, y, z, now))
            # Drop old points beyond fade window and also limit length
            cutoff = now - self.trail_fade_seconds
            trail[:] = [p for p in trail if p[3] >= cutoff]
            if len(trail) > self.trail_length:
                del trail[0:len(trail)-self.trail_length]
            # Draw faded segments with alpha based on age
            for i in range(1, len(trail)):
                x0, y0, z0, t0 = trail[i-1]
                x1, y1, z1, t1 = trail[i]
                age = now - t1
                alpha = max(0.0, min(1.0, 1.0 - age / self.trail_fade_seconds))
                rgba = mcolors.to_rgba(color, alpha)
                self.ax.plot([x0, x1], [y0, y1], [z0, z1], color=rgba, linewidth=2)

        # Spheres (e.g., smoke clouds)
        for sph in data.get('spheres', []) or []:
            cx, cy, cz = sph.get('x', 0.0), sph.get('y', 0.0), sph.get('z', 0.0)
            r = sph.get('radius', 10.0)
            color = sph.get('color', 'gray')
            alpha = sph.get('alpha', 0.3)
            # Coarse mesh to keep performance reasonable
            u = np.linspace(0, 2 * math.pi, 24)
            v = np.linspace(0, math.pi, 16)
            x = cx + r * np.outer(np.cos(u), np.sin(v))
            y = cy + r * np.outer(np.sin(u), np.sin(v))
            z = cz + r * np.outer(np.ones_like(u), np.cos(v))
            self.ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, antialiased=False)

        # Cylinders (e.g., true target)
        for cyl in data.get('cylinders', []) or []:
            cx, cy, cz = cyl.get('x', 0.0), cyl.get('y', 0.0), cyl.get('z', 0.0)
            h = cyl.get('height', 10.0)
            r = cyl.get('radius', 5.0)
            color = cyl.get('color', 'r')
            alpha = cyl.get('alpha', 0.5)
            theta = np.linspace(0, 2 * math.pi, 48)
            z_vals = np.linspace(cz, cz + h, 2)
            theta_grid, z_grid = np.meshgrid(theta, z_vals)
            x_grid = cx + r * np.cos(theta_grid)
            y_grid = cy + r * np.sin(theta_grid)
            self.ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, linewidth=0)
            # Top and bottom circles (wire)
            xt = cx + r * np.cos(theta)
            yt = cy + r * np.sin(theta)
            self.ax.plot(xt, yt, np.full_like(theta, cz), color=color, alpha=alpha)
            self.ax.plot(xt, yt, np.full_like(theta, cz + h), color=color, alpha=alpha)

        # Coordinate axes at origin
        ox, oy, oz = (data.get('axes_origin') or (0, 0, 0))
        self.ax.quiver(ox, oy, oz, 100, 0, 0, color='r')
        self.ax.quiver(ox, oy, oz, 0, 100, 0, color='g')
        self.ax.quiver(ox, oy, oz, 0, 0, 100, color='b')

        # Legend & unit info at top-right
        try:
            if self._legend is not None:
                self._legend.remove()
        except Exception:
            pass
        handles = [
            mpatches.Patch(color='r', label='Missile (red)'),
            mpatches.Patch(color='b', label='UAV (blue)'),
            mpatches.Patch(color='orange', label='True target / cylinder (orange)'),
            mpatches.Patch(color='yellow', label='Decoy / origin (yellow)'),
            mpatches.Patch(color='gray', label='Smoke cloud (gray)'),
        ]
        # Place legend at top-left to avoid overlapping with info text
        self._legend = self.ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.0, 1.0), framealpha=0.9)

        try:
            if self._info_text is not None:
                self._info_text.remove()
        except Exception:
            pass
        info = 'Units\nPos: meter (m)\nSpeed: m/s\nTime: second (s)'
        self._info_text = self.ax.text2D(0.99, 0.88, info, transform=self.ax.transAxes, ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', fc='white', alpha=0.6))

        plt.pause(self.pause_sec)


