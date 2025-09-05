# -*- coding: utf-8 -*-
import pygame
from typing import Dict, List, Tuple, Optional
try:
    # When used as a package
    from . import constants as C
    from .assets import load_graphics, load_sound, get_default_assets_paths
    from .hud import InfoHUD
    from .sprites import OrientedSprite, BasicSprite
except Exception:
    # When run as loose scripts (no package parent)
    import constants as C
    from assets import load_graphics, load_sound, get_default_assets_paths
    from hud import InfoHUD
    from sprites import OrientedSprite, BasicSprite


class Renderer:
    """Standalone Pygame renderer for UAV scenes.

    Usage:
        rnd = Renderer()
        rnd.init()
        while running:
            rnd.poll_events()
            rnd.begin_frame()
            rnd.draw_scene(render_data)
            rnd.end_frame()
    """

    def __init__(self, window_title: str = "UAV 可视化", images_dir: Optional[str] = None, music_dir: Optional[str] = None):
        self.window_title = window_title
        self.images_dir = images_dir
        self.music_dir = music_dir
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.graphics = {}
        self.sounds = {}
        self.hud: Optional[InfoHUD] = None
        self._sprites_created = False
        self._hero_sprites: List[OrientedSprite] = []
        self._enemy_sprites: List[OrientedSprite] = []
        self._goal_sprite: Optional[BasicSprite] = None
        self._obstacle_sprite: Optional[BasicSprite] = None
        self._trajectory_points: List[Tuple[float, float]] = []
        self._enemy_trajectories: List[List[Tuple[float, float]]] = []

    def init(self):
        pygame.init()
        try:
            pygame.font.init()
        except Exception:
            pass
        try:
            pygame.mixer.init()
        except pygame.error:
            pass
        self.screen = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))
        pygame.display.set_caption(self.window_title)
        self.clock = pygame.time.Clock()
        # Initialize HUD after pygame/font init
        try:
            self.hud = InfoHUD(title=self.window_title)
        except Exception:
            self.hud = None
        if self.images_dir is None or self.music_dir is None:
            default_images, default_music = get_default_assets_paths()
            self.images_dir = self.images_dir or default_images
            self.music_dir = self.music_dir or default_music
        if self.images_dir:
            self.graphics = load_graphics(self.images_dir)
        if self.music_dir:
            self.sounds = load_sound(self.music_dir)

    def poll_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def _ensure_sprites(self, hero_num: int, enemy_num: int):
        if self._sprites_created:
            return
        hero_img = self.graphics.get("fighter-blue")
        enemy_img = self.graphics.get("fighter-green")
        goal_img = self.graphics.get("goal")
        hole_img = self.graphics.get("hole")
        for _ in range(hero_num):
            self._hero_sprites.append(OrientedSprite(hero_img, size=(20, 20)))
        for _ in range(enemy_num):
            self._enemy_sprites.append(OrientedSprite(enemy_img, size=(20, 20)))
        self._goal_sprite = BasicSprite(goal_img, size=(20, 20))
        self._obstacle_sprite = BasicSprite(hole_img, size=(40, 40))
        self._sprites_created = True

    def begin_frame(self):
        assert self.screen is not None
        self.screen.fill((0, 0, 0))
        # draw enemy rectangle region
        pygame.draw.rect(self.screen, C.BLACK, C.ENEMY_AREA, 3)

    def draw_scene(self, data: Dict):
        """Draw the scene from a neutral data dict.

        Expected keys:
          - hero: List[Dict{x,y,theta}]
          - enemies: List[Dict{x,y,theta}]
          - goal: Dict{x,y}
          - obstacle: Dict{x,y}
          - trajectory: List[Tuple[x,y]]  (optional)
          - enemy_trajectories: List[List[Tuple[x,y]]] (optional)
          - metrics: Dict[str, float] (optional)
        """
        heroes = data.get("hero", [])
        enemies = data.get("enemies", [])
        self._ensure_sprites(len(heroes), len(enemies))

        # goal and obstacle
        goal = data.get("goal")
        if goal and self._goal_sprite:
            self._goal_sprite.set_pose(goal["x"], goal["y"]) 
            self.screen.blit(self._goal_sprite.image, self._goal_sprite.rect)
            pygame.draw.circle(self.screen, C.RED, (int(goal["x"]), int(goal["y"])), 40, 1)

        obstacle = data.get("obstacle")
        if obstacle and self._obstacle_sprite:
            self._obstacle_sprite.set_pose(obstacle["x"], obstacle["y"]) 
            self.screen.blit(self._obstacle_sprite.image, self._obstacle_sprite.rect)
            pygame.draw.circle(self.screen, C.BLACK, (int(obstacle["x"]), int(obstacle["y"])), 20, 1)

        # trajectories
        traj = data.get("trajectory")
        if traj:
            for i in range(1, len(traj)):
                pygame.draw.line(self.screen, C.BLUE, traj[i - 1], traj[i])
        enemy_trajs = data.get("enemy_trajectories") or []
        for t in enemy_trajs:
            for i in range(1, len(t)):
                pygame.draw.line(self.screen, C.GREEN, t[i - 1], t[i])

        # agents
        for i, hero in enumerate(heroes):
            sprite = self._hero_sprites[i]
            sprite.set_pose(hero.get("x", 0), hero.get("y", 0), hero.get("theta"))
            self.screen.blit(sprite.image, sprite.rect)
        for i, enemy in enumerate(enemies):
            sprite = self._enemy_sprites[i]
            sprite.set_pose(enemy.get("x", 0), enemy.get("y", 0), enemy.get("theta"))
            self.screen.blit(sprite.image, sprite.rect)

        metrics = data.get("metrics", {})
        if self.hud is not None:
            self.hud.update(metrics)
            self.hud.draw(self.screen)

    def end_frame(self):
        pygame.display.update()
        if self.clock:
            self.clock.tick(C.FPS)


