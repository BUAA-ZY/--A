# -*- coding: utf-8 -*-
import math
import pygame
from typing import Optional, Tuple


class BasicSprite(pygame.sprite.Sprite):
    """A lightweight sprite that holds a rect and optional image.

    Use set_pose(x, y) to position it. If image is provided, it will be blitted.
    """

    def __init__(self, image: Optional[pygame.Surface] = None, size: Tuple[int, int] = (20, 20)):
        super().__init__()
        if image is None:
            self.image = pygame.Surface(size, pygame.SRCALPHA)
            pygame.draw.circle(self.image, (0, 0, 255), (size[0] // 2, size[1] // 2), min(size) // 2)
            self.rect = self.image.get_rect()
        else:
            self.image = pygame.transform.scale(image, size)
            self.rect = self.image.get_rect()
        self.pos = (0.0, 0.0)

    def set_pose(self, x: float, y: float):
        self.pos = (x, y)
        self.rect.center = (x, y)


class OrientedSprite(BasicSprite):
    """Sprite that can be rotated by heading theta (radians)."""

    def __init__(self, image: Optional[pygame.Surface] = None, size: Tuple[int, int] = (20, 20)):
        super().__init__(image=image, size=size)
        self._orig_image = self.image.copy()
        self.theta = 0.0

    def set_pose(self, x: float, y: float, theta: Optional[float] = None):
        if theta is not None:
            self.theta = theta
        super().set_pose(x, y)
        self._apply_rotation()

    def _apply_rotation(self):
        angle_deg = self.theta * 57.29577951308232
        self.image = pygame.transform.rotozoom(self._orig_image, angle_deg, 1)
        self.rect = self.image.get_rect(center=self.rect.center)


