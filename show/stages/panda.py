"""Stage 4 - How researchers do it: the simulated attack (Sharif et al., 2016).

The glasses are drawn on the screen only; the model never sees them, and the
"Panda" label is fixed. The screen says so with a SIMULATION tag.

Key E shows how such glasses are made, step by step (like stage 1), next to an animation
of a glasses pattern that changes while a score bar moves. That animation is not computed
from a model, so it carries an ILLUSTRATION tag.
"""
import math

import numpy as np
import pygame
from PIL import Image

from .. import config
from ..draw import CYAN, GREEN, RED, WHITE, YELLOW
from ..players import is_big
from ..texts import t
from .base import Stage

MAX_PANDAS = 4
STEP_TIME = 3.5        # explanation: seconds per step
ILLUS_TIME = 10.0      # the pattern animation runs this long, then holds
GRID_W, GRID_H = 16, 6


def _frame_cells():
    """Cells of a glasses frame in a 16 x 6 grid: two lens rims and a bridge."""
    cells = []
    for c0 in (0, 9):
        for row in range(GRID_H):
            for col in range(c0, c0 + 7):
                if row in (0, GRID_H - 1) or col in (c0, c0 + 6):
                    cells.append((row, col))
    cells += [(1, 7), (1, 8)]
    return cells


FRAME_CELLS = _frame_cells()
_rng = np.random.default_rng(2016)
TARGET_COLORS = _rng.integers(0, 256, size=(len(FRAME_CELLS), 3))  # fixed, so each run looks the same


def load_glasses():
    img = np.array(Image.open(config.GLASSES_FILE).convert("RGBA"))
    white = (img[:, :, 0] > 245) & (img[:, :, 1] > 245) & (img[:, :, 2] > 245)
    img[white, 3] = 0  # white background -> transparent
    h, w = img.shape[:2]
    return pygame.image.frombuffer(np.ascontiguousarray(img).tobytes(), (w, h), "RGBA").copy()


def eye_points(face):
    """Left and right eye in frame pixels; estimated from the box when the keypoints are unsure."""
    k = face.kpts
    # keypoints are (x, y) or (x, y, confidence), depending on the model
    sure = k is not None and len(k) >= 2 and (k.shape[1] < 3 or (k[0][2] >= 0.5 and k[1][2] >= 0.5))
    if sure and min(k[0][0], k[1][0]) > 0:
        a, b = (k[0][0], k[0][1]), (k[1][0], k[1][1])
    else:
        x1, y1, x2, y2 = face.box
        w, h = x2 - x1, y2 - y1
        a, b = (x1 + 0.32 * w, y1 + 0.40 * h), (x1 + 0.68 * w, y1 + 0.40 * h)
    return (a, b) if a[0] <= b[0] else (b, a)


class PandaStage(Stage):
    name = "4 simulated attack"
    title_key = "s4_title"
    sub_key = "s4_sub"

    def __init__(self):
        self.glasses = None

    def enter(self, app):
        if self.glasses is None:
            self.glasses = load_glasses()
        self.explain = False         # the "how is it made" explanation (key E)
        self.explain_start = 0.0

    def on_key(self, key, app):
        if key == pygame.K_e:
            self.explain = not self.explain
            self.explain_start = app.now
            return True
        return False

    def targets(self, det):
        faces = [f for f in det.faces if f.conf >= config.FACE_VISIBLE_CONF]
        faces.sort(key=lambda f: -f.width)
        big = [f for f in faces if is_big(f, det.frame_w)]
        return (big or faces[:1])[:MAX_PANDAS]

    def draw(self, r, app, det, now):
        for f in self.targets(det):
            (ax, ay), (bx, by) = [r.point(*p) for p in eye_points(f)]
            dist = math.hypot(bx - ax, by - ay)
            if dist > 2:
                angle = -math.degrees(math.atan2(by - ay, bx - ax))
                scale = dist * config.GLASSES_SCALE / self.glasses.get_width()
                img = pygame.transform.rotozoom(self.glasses, angle, scale)
                r.screen.blit(img, img.get_rect(center=((ax + bx) // 2, (ay + by) // 2)))
            box = r.labeled_box(f.box, f"{t('panda')} {f.conf:.0%}", RED)
            r.text(t("simulation"), 16, YELLOW, midtop=(box.centerx, box.bottom + r.u(6)))

        sw, sh = r.size
        if self.explain:
            self._draw_explain(r, now - self.explain_start)
            return

        # the story of the paper, on the right (the SIMULATION tag on each box always stays)
        if not app.overlays:
            return
        lines = t("s4_story")
        line = r.u(34)
        panel = pygame.Rect(sw - r.u(470), r.u(140), r.u(450), r.u(40) + line * len(lines))
        r.shade(panel, 180, r.u(14))
        for i, s in enumerate(lines):
            color = YELLOW if i >= 5 else WHITE
            r.text(s, 22, color, shadow=False, topleft=(panel.x + r.u(18), panel.y + r.u(20) + i * line))
        r.text("Sharif et al. (2016)", 16, (200, 200, 200), shadow=False, bold=False,
               bottomright=(panel.right - r.u(14), panel.bottom - r.u(8)))

    def _draw_explain(self, r, el):
        """Steps on the left, one by one; the pattern illustration on the right."""
        sw, sh = r.size
        r.shade(pygame.Rect(0, 0, sw, sh), 205)
        x, y, w = r.u(50), r.u(125), r.u(640)
        r.text(t("x4_title"), 34, CYAN, topleft=(x, y))
        y += r.u(56)
        steps = t("x4_steps")
        shown = min(len(steps), 1 + int(el / STEP_TIME))
        for i in range(shown):
            circle = (x + r.u(15), y + r.u(15))
            pygame.draw.circle(r.screen, YELLOW, circle, r.u(15))
            r.text(str(i + 1), 20, (0, 0, 0), shadow=False, center=circle)
            for line in r.wrap(steps[i], 24, w - r.u(44)):
                r.text(line, 24, WHITE, topleft=(x + r.u(44), y))
                y += r.u(32)
            y += r.u(14)
        if el >= STEP_TIME * len(steps):
            r.text(t("x4_end"), 26, YELLOW, topleft=(x, y + r.u(10)))

        if el >= STEP_TIME:  # the pattern starts to change with step 2
            self._draw_pattern(r, min(1.0, (el - STEP_TIME) / ILLUS_TIME), el)

    @staticmethod
    def _draw_pattern(r, k, el):
        """Illustration: the frame pixels move to a strange pattern, the scores move with them."""
        sw, _ = r.size
        cell = r.u(28)
        gx, gy = sw - r.u(40) - GRID_W * cell, r.u(190)
        flicker = 0.0 if k >= 1 else 0.25
        rng = np.random.default_rng(int(el * 8))         # a small change every update, like a step
        for (row, col), target in zip(FRAME_CELLS, TARGET_COLORS):
            c = (1 - k) * np.array([45, 45, 45]) + k * target
            c = np.clip(c + flicker * rng.integers(-60, 61, size=3), 0, 255).astype(int)
            rect = pygame.Rect(gx + col * cell, gy + row * cell, cell - 2, cell - 2)
            pygame.draw.rect(r.screen, tuple(c), rect)
        pygame.draw.rect(r.screen, WHITE, (gx - 6, gy - 6, GRID_W * cell + 10, GRID_H * cell + 10), 2,
                         border_radius=r.u(6))
        r.text(t("x4_step", n=int(k * 1000)), 22, WHITE, topleft=(gx, gy + GRID_H * cell + r.u(14)))

        # scores: the right answer falls, the wrong answer rises
        bw = GRID_W * cell
        by = gy + GRID_H * cell + r.u(60)
        s = k ** 0.7
        for label, value, color in ((t("x4_score_you"), 0.92 - 0.84 * s, GREEN),
                                    (f"Score: {t('panda')}", 0.03 + 0.82 * s, RED)):
            r.text(label, 20, WHITE, topleft=(gx, by))
            r.bar(pygame.Rect(gx, by + r.u(28), bw, r.u(22)), value, color)
            by += r.u(70)
        r.text(t("x4_illustration"), 16, YELLOW, topleft=(gx, by + r.u(4)))
