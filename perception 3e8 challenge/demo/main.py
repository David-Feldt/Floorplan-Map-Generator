"""
Delivery Robot — Potential Field Navigation

Usage:
    python main.py                # full system (camera + LLM)
    python main.py --no-llm       # camera only, no LLM context
"""
from __future__ import annotations
import sys
import threading

import pygame

from config import FPS
from force_field import ForceField
from perception import Camera, perception_loop
from ui import Renderer


def main():
    no_llm = "--no-llm" in sys.argv

    # --- shared state ---
    shared: dict = {"scene": None}
    stop = threading.Event()

    force_field = ForceField()
    renderer = Renderer()

    # --- perception ---
    camera = Camera()
    t_percep = threading.Thread(
        target=perception_loop, args=(shared, camera, stop), daemon=True
    )
    t_percep.start()

    # --- LLM context (optional) ---
    if not no_llm:
        try:
            from llm_context import llm_loop

            t_llm = threading.Thread(
                target=llm_loop, args=(shared, force_field, stop), daemon=True
            )
            t_llm.start()
        except ImportError:
            print("[warn] llm_context not available, running without LLM")

    # --- render loop ---
    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        scene = shared.get("scene")
        if scene:
            force_field.compute(scene, dt)

        renderer.draw(force_field, scene)

    stop.set()
    pygame.quit()


if __name__ == "__main__":
    main()
