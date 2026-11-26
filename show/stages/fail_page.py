"""Stage 3 explanation (key E): why can the AI miss a face? Live, with real numbers.

The page follows the explanation face (facelock.py: it never switches by itself; TAB picks
another one) and shows its live face score from the modern AI (the face model of the show).
When the face is covered, the page keeps its last place, so the effect stays visible; the score
then reads "not found" (the live model reports faces only from config.FACE_DETECT_CONF up).

Detailed (Shift+E): also three measurements of the face area in the camera image: its width in
pixels, how light it is (mean gray value, 0-255) and its contrast (standard deviation of the
gray values). For covering and turning, the page claims no cause: it only shows the score.
"""
import cv2
import numpy as np
import pygame

from .. import config
from ..draw import CYAN, GREY, WHITE, YELLOW
from ..texts import t
from ..vision import crop_square
from .page_parts import blit_image, draw_steps, score_bar

SIZE_FULL = 250        # face width in pixels that fills the size bar
CONTRAST_FULL = 70     # standard deviation that fills the contrast bar


def measure(frame, box):
    """Width in pixels, mean gray value and contrast of the box area of the frame (BGR)."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float64)
    return {"width": x2 - x1, "light": float(gray.mean()), "contrast": float(gray.std())}


class FailExplainer:
    def __init__(self):
        self.on = False
        self.box = self.score = self.numbers = self.crop = None

    def open(self, now):
        self.on = True
        self.box = self.score = self.numbers = self.crop = None

    def close(self):
        self.on = False

    def toggle(self, now):
        if self.on:
            self.close()
        else:
            self.open(now)

    def update(self, app, det, now):
        if not self.on or app.frame is None:
            return
        self.box, self.score = app.face_lock.box, app.face_lock.conf
        if self.box is not None:
            self.numbers = measure(app.frame, self.box)
            crop = crop_square(app.frame, self.box, margin=0.35)
            self.crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.size else None

    # --- drawing ---
    def draw(self, r, app, now):
        sw, sh = r.size
        r.shade(pygame.Rect(0, 0, sw, sh), 235)
        detail = config.EXPLAIN_DETAIL
        draw_steps(r, t("xf_title"), t("xf_steps" if detail else "xfs_steps"),
                   r.u(40), r.top + r.u(24), r.u(500), size=21 if detail else 23)
        r.text(t("x3_close"), 18, GREY, bottomleft=(r.u(40), sh - r.u(20)))

        area = pygame.Rect(sw - r.u(680), r.top + r.u(30), r.u(650), sh - r.top - r.u(60))
        if self.box is None or self.numbers is None:
            r.text(t("x3_no_face"), 28, YELLOW, center=area.center)
            return
        size = r.u(250 if detail else 300)
        if self.crop is not None:
            blit_image(r, self.crop, pygame.Rect(area.x, area.y, size, size))
        pygame.draw.rect(r.screen, WHITE, (area.x - 2, area.y - 2, size + 4, size + 4), 2)

        sx = area.x + size + r.u(30)
        r.text(t("xf_score"), 22, WHITE, topleft=(sx, area.y))
        bar = pygame.Rect(sx, area.y + r.u(40), area.right - sx, r.u(40))
        score_bar(r, bar, self.score, config.FACE_VISIBLE_CONF)
        yy = bar.bottom + r.u(14)
        for line in r.wrap(t("xf_limit", p=f"{config.FACE_VISIBLE_CONF:.0%}"), 16, bar.w, bold=False):
            r.text(line, 16, GREY, bold=False, topleft=(sx, yy))
            yy += r.u(21)
        if not detail:
            return

        # three measurements of the face area: one neutral color, the numbers tell the value
        n = self.numbers
        rows = [(t("xf_size", n=n["width"]), n["width"] / SIZE_FULL),
                (t("xf_light", v=f"{n['light']:.0f}"), n["light"] / 255),
                (t("xf_contrast", v=f"{n['contrast']:.0f}"), n["contrast"] / CONTRAST_FULL)]
        my = area.y + size + r.u(36)
        for label, frac in rows:
            r.text(label, 20, WHITE, topleft=(area.x, my))
            r.bar(pygame.Rect(area.x + r.u(330), my + r.u(3), area.w - r.u(330), r.u(20)), min(1.0, frac), CYAN)
            my += r.u(42)
        r.text(t("xf_real"), 16, CYAN, bold=False, topleft=(area.x, my + r.u(4)))
