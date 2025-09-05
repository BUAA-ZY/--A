# -*- coding: utf-8 -*-
import os
import pygame
from typing import Dict, Tuple


def _maybe_convert(surface: pygame.Surface) -> pygame.Surface:
    """Convert surface for faster blitting if a display surface exists."""
    try:
        if pygame.display.get_init() and pygame.display.get_surface() is not None:
            if surface.get_alpha() is not None:
                return surface.convert_alpha()
            return surface.convert()
    except pygame.error:
        pass
    return surface


def load_graphics(path: str, accept=(".jpg", ".png", ".bmp", ".gif")) -> Dict[str, pygame.Surface]:
    """Load all images under a directory into a dict {name: surface}.

    - path: assets root for images
    - accept: tuple of file extensions
    """
    graphics: Dict[str, pygame.Surface] = {}
    if not os.path.isdir(path):
        return graphics
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext.lower() in accept:
            surface = pygame.image.load(os.path.join(path, filename))
            graphics[name] = _maybe_convert(surface)
    return graphics


def load_sound(path: str, accept=(".wav", ".mp3")) -> Dict[str, pygame.mixer.Sound]:
    """Load all sounds under a directory into a dict {name: sound}."""
    sounds: Dict[str, pygame.mixer.Sound] = {}
    if not os.path.isdir(path):
        return sounds
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext.lower() in accept:
            try:
                sounds[name] = pygame.mixer.Sound(os.path.join(path, filename))
            except pygame.error:
                # Mixer may not be initialized or format unsupported; skip silently
                continue
    return sounds


def get_default_assets_paths() -> Tuple[str, str]:
    """Return default (images_dir, music_dir) using repository relative paths.

    Falls back to empty strings if not found.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.normpath(os.path.join(this_dir, "..", "assignment", "source", "image"))
    music_dir = os.path.normpath(os.path.join(this_dir, "..", "assignment", "source", "music"))
    if not os.path.isdir(images_dir):
        images_dir = ""
    if not os.path.isdir(music_dir):
        music_dir = ""
    return images_dir, music_dir


