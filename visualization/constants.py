# -*- coding: utf-8 -*-
import pygame

# Screen and world layout
SCREEN_W, SCREEN_H = 1000, 800
SCREEN_SIZE = (SCREEN_W, SCREEN_H)

# Enemy area rectangle (x, y, width, height)
ENEMY_AREA_X, ENEMY_AREA_Y, ENEMY_AREA_WITH, ENEMY_AREA_HEIGHT = 50, 50, 700, 600
ENEMY_AREA = (ENEMY_AREA_X, ENEMY_AREA_Y, ENEMY_AREA_WITH, ENEMY_AREA_HEIGHT)

# Timing
FPS = 60

# Fonts
FONT_CHINESE = '华文新魏'
FONT_ENGLISH = 'arial'

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 128, 0)

# Events and flags
CREATE_ENEMY_EVENT = pygame.USEREVENT
CLICK = False
OPEN_MENU = False
OPEN_MUSIC = True
OPEN_SOUND = True

# Others
BULLET_SPEED = 5


