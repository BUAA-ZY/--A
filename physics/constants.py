# -*- coding: utf-8 -*-
"""Physical constants and default parameters for the A-Problem simulation."""

# Gravity (m/s^2)
G_ACCEL = 9.8

# Missile speed (m/s)
MISSILE_SPEED = 300.0

# UAV speed bounds (m/s)
UAV_SPEED_MIN = 70.0
UAV_SPEED_MAX = 140.0

# Smoke cloud properties
CLOUD_RADIUS = 10.0  # meters
CLOUD_SINK_RATE = 3.0  # m/s downward after detonation
CLOUD_EFFECTIVE_SECONDS = 20.0  # seconds of effective obscuration

# Numerical settings
DEFAULT_DT = 0.02  # s, simulation step
EPS = 1e-9


