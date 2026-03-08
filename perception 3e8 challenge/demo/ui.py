from __future__ import annotations
import pygame
import numpy as np
import cv2

from config import WINDOW_WIDTH, WINDOW_HEIGHT, ROBOT_SIZE
from force_field import ForceField
from perception import SceneFrame


# colours
BG = (20, 20, 30)
WHITE = (255, 255, 255)
GREY = (180, 180, 180)
DARK_GREY = (60, 60, 70)
GREEN = (80, 200, 80)
YELLOW = (200, 200, 80)
RED = (200, 80, 80)
BLUE = (80, 130, 220)
ARROW_GOAL = (80, 200, 80)
ARROW_REPEL = (200, 80, 80)
ARROW_NET = WHITE


class Renderer:
    # panel layout
    DEPTH_W = 320          # left panel (depth feed, scaled down from 640)
    LOG_W = 240            # right panel
    PANEL_PAD = 10

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Delivery Robot — Force Field Demo")
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_lg = pygame.font.SysFont("monospace", 18, bold=True)
        self.log: list[str] = []
        self._last_action = ""

    # ------------------------------------------------------------------
    # Main draw call
    # ------------------------------------------------------------------

    def draw(self, ff: ForceField, scene: SceneFrame | None):
        self.screen.fill(BG)
        self._draw_depth_panel(scene)
        self._draw_force_panel(ff, scene)
        self._draw_log_panel()
        pygame.display.flip()

    # ------------------------------------------------------------------
    # Left panel: depth feed + overlay
    # ------------------------------------------------------------------

    def _draw_depth_panel(self, scene: SceneFrame | None):
        x0 = self.PANEL_PAD
        y0 = self.PANEL_PAD
        w = self.DEPTH_W
        h = WINDOW_HEIGHT - 2 * self.PANEL_PAD

        # border
        pygame.draw.rect(self.screen, DARK_GREY, (x0, y0, w, h), 1)

        if scene is None or scene.depth_colorized is None:
            lbl = self.font.render("No camera feed", True, GREY)
            self.screen.blit(lbl, (x0 + 10, y0 + h // 2))
            return

        # scale depth visualisation to panel size
        vis = scene.depth_colorized
        vis_resized = cv2.resize(vis, (w, h))
        # draw bounding boxes on vis
        scale_x = w / vis.shape[1]
        scale_y = h / vis.shape[0]
        for obj in scene.objects:
            cx, cy, bw, bh = obj.bbox
            rx = int((cx - bw // 2) * scale_x)
            ry = int((cy - bh // 2) * scale_y)
            rw = int(bw * scale_x)
            rh = int(bh * scale_y)
            color = RED if obj.pose in ("arms_raised", "on_ground") else GREEN
            cv2.rectangle(vis_resized, (rx, ry), (rx + rw, ry + rh), color, 2)
            label = f"{obj.obj_class} {obj.distance_m:.1f}m"
            cv2.putText(vis_resized, label, (rx, ry - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # convert BGR → RGB and blit
        surf = pygame.surfarray.make_surface(
            cv2.cvtColor(vis_resized, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        )
        self.screen.blit(surf, (x0, y0))

    # ------------------------------------------------------------------
    # Centre panel: force field + robot
    # ------------------------------------------------------------------

    def _draw_force_panel(self, ff: ForceField, scene: SceneFrame | None):
        x0 = self.DEPTH_W + 2 * self.PANEL_PAD
        w = WINDOW_WIDTH - self.DEPTH_W - self.LOG_W - 4 * self.PANEL_PAD
        cy = WINDOW_HEIGHT // 2

        # centre line (track the robot moves along)
        pygame.draw.line(self.screen, DARK_GREY, (x0, cy), (x0 + w, cy), 1)

        # robot position mapped to panel
        robot_x = x0 + w // 2 + int(ff.position * 60)
        robot_x = max(x0 + ROBOT_SIZE, min(x0 + w - ROBOT_SIZE, robot_x))

        # colour by velocity
        if ff.velocity > 0.5:
            color = GREEN
        elif ff.velocity < -0.5:
            color = RED
        else:
            color = YELLOW

        robot_rect = pygame.Rect(
            robot_x - ROBOT_SIZE // 2,
            cy - ROBOT_SIZE // 2,
            ROBOT_SIZE,
            ROBOT_SIZE,
        )
        pygame.draw.rect(self.screen, color, robot_rect, border_radius=6)

        # action label ON the square
        action = ff.get_display_action()
        lbl = self.font.render(action, True, WHITE)
        self.screen.blit(lbl, (
            robot_rect.centerx - lbl.get_width() // 2,
            robot_rect.centery - lbl.get_height() // 2,
        ))

        # reason BELOW the square
        reason = ff.get_display_reason()
        rlbl = self.font.render(reason, True, GREY)
        self.screen.blit(rlbl, (
            robot_rect.centerx - rlbl.get_width() // 2,
            robot_rect.bottom + 8,
        ))

        # net force arrow above the square
        arrow_len = int(ff.last_net_force * 40)
        arrow_y = robot_rect.top - 16
        pygame.draw.line(
            self.screen, ARROW_NET,
            (robot_rect.centerx, arrow_y),
            (robot_rect.centerx + arrow_len, arrow_y),
            3,
        )
        # arrowhead
        if abs(arrow_len) > 4:
            tip_x = robot_rect.centerx + arrow_len
            sign = 1 if arrow_len > 0 else -1
            pygame.draw.polygon(self.screen, ARROW_NET, [
                (tip_x, arrow_y),
                (tip_x - sign * 6, arrow_y - 4),
                (tip_x - sign * 6, arrow_y + 4),
            ])

        # per-object force indicators (small coloured dots along the track)
        if scene:
            for obj in scene.objects:
                obj_screen_x = robot_x + int(obj.distance_m * 60)
                obj_screen_x = max(x0, min(x0 + w, obj_screen_x))
                dot_color = RED if obj.pose in ("arms_raised", "on_ground") else BLUE
                pygame.draw.circle(self.screen, dot_color, (obj_screen_x, cy), 6)
                olbl = self.font.render(
                    f"{obj.obj_class} {obj.distance_m:.1f}m", True, dot_color
                )
                self.screen.blit(olbl, (obj_screen_x - olbl.get_width() // 2, cy + 12))

        # velocity readout
        vlbl = self.font.render(f"vel: {ff.velocity:+.2f}", True, GREY)
        self.screen.blit(vlbl, (x0, WINDOW_HEIGHT - 30))

        # update log
        if action and action != self._last_action:
            self.log.append(f"{action}: {reason}")
            if len(self.log) > 30:
                self.log.pop(0)
            self._last_action = action

    # ------------------------------------------------------------------
    # Right panel: decision log
    # ------------------------------------------------------------------

    def _draw_log_panel(self):
        x0 = WINDOW_WIDTH - self.LOG_W - self.PANEL_PAD
        y0 = self.PANEL_PAD
        w = self.LOG_W
        h = WINDOW_HEIGHT - 2 * self.PANEL_PAD

        pygame.draw.rect(self.screen, DARK_GREY, (x0, y0, w, h), 1)

        title = self.font_lg.render("Decision Log", True, WHITE)
        self.screen.blit(title, (x0 + 8, y0 + 6))

        y = y0 + 30
        for entry in reversed(self.log[-20:]):
            if y > y0 + h - 16:
                break
            # truncate long lines
            text = entry[:34] if len(entry) > 34 else entry
            surf = self.font.render(text, True, GREY)
            self.screen.blit(surf, (x0 + 8, y))
            y += 16
