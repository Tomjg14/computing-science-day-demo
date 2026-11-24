"""Stage 4 - How researchers do it: the simulated attack (Sharif et al., 2016).

The glasses are drawn on the screen only; the model never sees them, and the
"Panda" label is fixed. The screen says so with a SIMULATION tag.

Key E shows how such glasses are made, step by step (like stage 1). On the right, the face of
the player shows as pixel blocks (one camera image, taken when E is pressed and kept in memory
only). Then a glasses frame goes on top, the face fades to the background, and the frame
pixels change into a pattern while two score bars move; a magnifier shows the numbers of one
frame pixel before and now. The pattern and the scores are not computed from a model, so
they carry an ILLUSTRATION tag.
"""
import math

import cv2
import numpy as np
import pygame
from PIL import Image

from .. import config
from ..draw import CYAN, GREEN, RED, WHITE, YELLOW
from ..players import is_big
from ..texts import t
from ..vision import crop_square
from .base import Stage

MAX_PANDAS = 4
STEP_TIME = 3.5        # explanation: seconds per step
ILLUS_TIME = 10.0      # the pattern animation runs this long, then holds
GRID_W, GRID_H = 16, 6     # the glasses frame
FACE_N = 16                # the face: FACE_N x FACE_N pixel blocks
FACE_MARGIN = 0.05         # space around the face box in the face image (small: the glasses span the face)
FRAME_START = (45, 45, 45)  # color of the plain frame before the change
MAGNIFY_TIME = 1.5         # seconds per pixel in the magnifier


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
            if self.explain:
                self._take_face(app)
            return True
        return False

    def _take_face(self, app):
        """One image of the largest face as FACE_N x FACE_N color blocks, and the row of the eyes.
        No face: gray blocks, eyes at the usual height."""
        self.face_cells, self.frame_row = np.full((FACE_N, FACE_N, 3), 110, np.uint8), 5
        faces = self.targets(app.det) if app.det is not None else []
        if not faces or app.frame is None:
            return
        f = faces[0]
        crop = crop_square(app.frame, f.box, margin=FACE_MARGIN)
        if crop.size == 0:
            return
        small = cv2.resize(crop, (FACE_N, FACE_N), interpolation=cv2.INTER_AREA)
        self.face_cells = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        # the frame goes over the eyes: their height in the face image, as a block row
        x1, y1, x2, y2 = f.box
        half = max(x2 - x1, y2 - y1) * (0.5 + FACE_MARGIN)
        top = max(0.0, (y1 + y2) / 2 - half)
        eye_y = sum(pt[1] for pt in eye_points(f)) / 2
        eye_row = (eye_y - top) / (2 * half) * FACE_N   # in blocks, from the top
        self.frame_row = int(np.clip(round(eye_row - GRID_H / 2), 0, FACE_N - GRID_H))

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
        """Steps on the left, one by one; the face and the glasses pattern on the right."""
        sw, sh = r.size
        r.shade(pygame.Rect(0, 0, sw, sh), 205)
        x, y, w = r.u(50), r.top + r.u(24), r.u(620)
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
        self._draw_face_pattern(r, el)

    def _draw_face_pattern(self, r, el):
        """Step 1: the face as pixel blocks. Step 2: a plain glasses frame on top, the face fades.
        From step 3: the frame pixels change into a pattern; the scores move with them."""
        sw, _ = r.size
        cell = r.u(24)
        gx, gy = sw - r.u(40) - FACE_N * cell, r.top + r.u(20)
        glasses = el >= STEP_TIME                                      # from step 2
        k = min(1.0, max(0.0, (el - 2 * STEP_TIME) / ILLUS_TIME))      # pattern progress, from step 3
        face = self.face_cells.astype(float)
        if glasses:
            face = face * 0.35 + 15                                     # the face fades to the background
        for row in range(FACE_N):
            for col in range(FACE_N):
                pygame.draw.rect(r.screen, tuple(int(v) for v in face[row, col]),
                                 (gx + col * cell, gy + row * cell, cell - 1, cell - 1))
        colors = []
        if glasses:
            flicker = 0.25 if 0 < k < 1 else 0.0
            rng = np.random.default_rng(int(el * 8))                    # a small change every update, like a step
            for (row, col), target in zip(FRAME_CELLS, TARGET_COLORS):
                c = (1 - k) * np.array(FRAME_START) + k * target
                c = np.clip(c + flicker * rng.integers(-60, 61, size=3), 0, 255).astype(int)
                colors.append(tuple(int(v) for v in c))
                rect = pygame.Rect(gx + col * cell, gy + (self.frame_row + row) * cell, cell - 1, cell - 1)
                pygame.draw.rect(r.screen, colors[-1], rect)
        pygame.draw.rect(r.screen, WHITE, (gx - 5, gy - 5, FACE_N * cell + 9, FACE_N * cell + 9), 2,
                         border_radius=r.u(6))
        y = gy + FACE_N * cell + r.u(10)
        if k > 0:
            r.text(t("x4_step", n=int(k * 1000)), 18, WHITE, topleft=(gx, y))

        # magnifier: one pixel and its three numbers (step 1: a face pixel; later: a frame pixel, before -> now)
        y += r.u(28)
        n = int(el / MAGNIFY_TIME)
        if not glasses:
            row, col = self.frame_row + 1, (3, 12, 7)[n % 3]
            rgb = tuple(int(v) for v in self.face_cells[row, col])
            line = t("x4_pixel", r=rgb[0], g=rgb[1], b=rgb[2])
        else:
            i = (n * 7) % len(FRAME_CELLS)
            row, col = FRAME_CELLS[i][0] + self.frame_row, FRAME_CELLS[i][1]
            rgb = colors[i]
            line = t("x4_change", r0=FRAME_START[0], g0=FRAME_START[1], b0=FRAME_START[2],
                     r=rgb[0], g=rgb[1], b=rgb[2])
        spot = pygame.Rect(gx + col * cell, gy + row * cell, cell - 1, cell - 1)
        pygame.draw.rect(r.screen, YELLOW, spot.inflate(r.u(6), r.u(6)), r.u(3))
        swatch = pygame.Rect(gx, y, r.u(34), r.u(34))
        pygame.draw.rect(r.screen, rgb, swatch)
        pygame.draw.rect(r.screen, WHITE, swatch, 2)
        r.text(line, 18, WHITE, midleft=(swatch.right + r.u(12), swatch.centery))

        # scores: the right answer falls, the wrong answer rises (from step 3)
        by = y + r.u(48)
        bw = FACE_N * cell
        s = k ** 0.7
        for label, value, color in ((t("x4_score_you"), 0.92 - 0.84 * s, GREEN),
                                    (f"Score: {t('panda')}", 0.03 + 0.82 * s, RED)):
            r.text(label, 18, WHITE, topleft=(gx, by))
            r.bar(pygame.Rect(gx + r.u(130), by, bw - r.u(130), r.u(20)), value, color)
            by += r.u(32)
        r.text(t("x4_illustration"), 16, YELLOW, topleft=(gx, by + r.u(2)))
