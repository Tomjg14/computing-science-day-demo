"""Chapters of the group lesson that the pass-by show does not have: the opening, the detector
explanations as a chapter of their own, the hiding game that starts with the page on why the AI
can miss a face, the traffic sign and the closing card.

The other chapters (pixels, detection, evasion game, glasses, Hit the number) are the same stage
objects as in the pass-by show, or live in their own file.
"""
import math
from statistics import median_low

import pygame

from .. import config
from ..draw import BLACK, GREY, WHITE, YELLOW
from ..players import box_area
from ..texts import t
from .base import Stage
from .detect import DetectStage
from .evasion import EvasionStage

SIGN_RED = (196, 30, 42)


class OpeningStage(Stage):
    """Everyone in view gets a faint box; the focus group (the players) gets a normal box.
    The count is the middle value of the last seconds, so it does not jump with every image."""
    name = "opening"
    title_key = "l1_title"
    sub_key = "l1_sub"

    def enter(self, app):
        self.counts = []   # (time, number of people) per detection update

    def update(self, app, det, now, dt):
        self.counts.append((now, len(det.persons)))
        self.counts = [c for c in self.counts if now - c[0] <= config.OPENING_COUNT_TIME]

    def count(self):
        return median_low([n for _, n in self.counts]) if self.counts else 0

    def draw(self, r, app, det, now):
        players = app.players.players
        others = sorted((o for o in det.persons if o.id not in players), key=lambda o: -box_area(o.box))
        for o in others[:config.OPENING_MAX_BOXES]:
            r.faint_box(o.box)
        for p in players.values():
            if p.body is None:
                continue
            box = p.face.box if p.face is not None else p.body
            r.labeled_box(box, t("face") if p.face is not None else t("person"), p.color, width=4, size=22)
        if not app.overlays:
            return
        sw, sh = r.size
        r.banner(sh - r.u(120), r.u(120))
        r.text(t("l1_count", n=self.count()), 56, WHITE, center=(sw // 2, sh - r.u(75)))
        r.text(t("l1_front", n=len(players)), 24, YELLOW, center=(sw // 2, sh - r.u(25)))


class DetectExplainStage(DetectStage):
    """The detection stage, but it starts with the explanation pages: the old AI, the modern AI,
    and from clues to boxes (E goes on, then closes them); then it is free detection with the
    same games."""
    name = "how a detector works"
    title_key = "l3_title"
    sub_key = "l3_sub"

    def enter(self, app):
        super().enter(app)
        self._open("old", app.now)


class HideStage(EvasionStage):
    """The hiding game, but it starts with the page on why the AI can miss a face.
    E closes it; then it is the same game."""
    name = "why the AI misses + hiding game"

    def enter(self, app):
        super().enter(app)
        self.explainer.open(app.now)


class SignStage(Stage):
    """Our own drawing of a stop sign with sticker blocks: the same idea as the glasses, on a
    traffic sign. An illustration only: no model looks at it, and the screen says so."""
    name = "traffic sign"
    title_key = "l8_title"
    sub_key = None

    def draw(self, r, app, det, now):
        sw, sh = r.size
        r.shade(pygame.Rect(0, 0, sw, sh), 200)
        cx, cy, rad = sw // 2, r.top + (sh - r.top) // 2 - r.u(10), r.u(230)
        _octagon(r.screen, WHITE, (cx, cy), rad)
        _octagon(r.screen, SIGN_RED, (cx, cy), int(rad * 0.9))
        r.text("STOP", 150, WHITE, shadow=False, center=(cx, cy))
        # sticker blocks: two above and two below the word, a little tilted
        for dx, dy, w, h, color, angle in ((-50, -140, 130, 32, WHITE, -8), (80, -138, 110, 32, BLACK, 6),
                                           (-75, 128, 110, 32, BLACK, 5), (70, 150, 130, 32, WHITE, -6)):
            block = pygame.Surface((r.u(w), r.u(h)), pygame.SRCALPHA)
            block.fill((*color, 255))
            block = pygame.transform.rotate(block, angle)
            k = r.u(100) / 100   # r.u() keeps sizes positive, so offsets scale here
            r.screen.blit(block, block.get_rect(center=(cx + int(dx * k), cy + int(dy * k))))
        r.text(t("illustration_tag"), 22, YELLOW, midtop=(cx, cy + rad + r.u(14)))
        r.text(config.SOURCE_SIGN, 18, GREY, shadow=False, bold=False,
               bottomright=(sw - r.u(20), sh - r.u(14)))


def _octagon(surface, color, center, radius):
    cx, cy = center
    pts = [(cx + radius * math.cos(math.radians(22.5 + 45 * k)),
            cy + radius * math.sin(math.radians(22.5 + 45 * k))) for k in range(8)]
    pygame.draw.polygon(surface, color, pts)


class ClosingStage(Stage):
    """Thank you, and a large QR code."""
    name = "closing"
    big_qr = True   # the small QR code in the corner stays away

    def draw(self, r, app, det, now):
        sw, sh = r.size
        r.shade(pygame.Rect(0, 0, sw, sh), 200)
        r.text(t("l9_thanks"), 72, WHITE, center=(sw // 2, r.u(110)))
        if app.qr is not None and app.show_qr:
            q = r.u(330)
            r.screen.blit(pygame.transform.smoothscale(app.qr, (q, q)), (sw // 2 - q // 2, r.u(190)))
            r.text(t("l9_qr"), 28, YELLOW, center=(sw // 2, r.u(190) + q + r.u(40)))
