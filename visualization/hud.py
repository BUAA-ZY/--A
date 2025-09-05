# -*- coding: utf-8 -*-
import pygame
from typing import Dict, Tuple
try:
    from . import constants as C
except Exception:
    import constants as C


class InfoHUD:
    """Simple HUD for showing basic information.

    Usage:
        hud = InfoHUD()
        hud.update({"episodes": 1, "wins": 0})
        hud.draw(screen)
    """

    def __init__(self, title: str = "可视化界面"):
        if not pygame.get_init():
            pygame.init()
        try:
            if not pygame.font.get_init():
                pygame.font.init()
        except Exception:
            pass
        self.title = title
        self.state_labels = []
        self.info_labels = []
        self._create_static_labels()
        self._create_info_labels()

    def _create_static_labels(self):
        self.state_labels = [
            (self._label(self.title, size=28), (300, 0)),
        ]

    def _create_info_labels(self):
        self.info_labels = [
            (self._label("通用信息", size=18), (0, 0))
        ]
        self.info_rect = self.info_labels[0][0].get_rect()

    def _label(self, text: str, size: int = 22, chinese: bool = True, color: Tuple[int, int, int] = C.WHITE) -> pygame.Surface:
        font_name = C.FONT_CHINESE if chinese else C.FONT_ENGLISH
        font = pygame.font.SysFont(font_name, size)
        return font.render(text, True, color)

    def update(self, metrics: Dict[str, float]):
        self.info_labels = [
            (self._label("通用信息", size=18), (0, 0)),
        ]
        y = 20
        for k, v in metrics.items():
            self.info_labels.append((self._label(f"{k}: {v}", size=16), (0, y)))
            y += 18

    def draw(self, surface: pygame.Surface):
        for img, pos in self.state_labels:
            surface.blit(img, pos)
        for img, pos in self.info_labels:
            surface.blit(img, pos)


