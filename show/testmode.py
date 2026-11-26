"""Test mode (key Y, or python -m show --test): a simulated group, to test the lesson alone.

The AI results of each update get config.TEST_FACES extra people in two rows, like a group in a
half circle: each person has a body box and a face. The operator "hides" faces one at a time
(, and .), as the group would; a person who hides the face keeps the body, as in the real game.
Shift+, and Shift+. remove or add a whole person (the opening count). Each visible face and body
is missed now and then (config.TEST_FLICKER), so the median counts get a real test. The real
people in view still count as well.

Simulated faces and bodies have ids from ID_BASE up, so they never mix with the ids of the
trackers. The "faces today" counter skips them. The screen always says that test mode is on.
"""
import math
import random
from dataclasses import replace

import pygame

from . import config
from .vision import Face, Obj

ID_BASE = 1_000_000          # far above the ids the face tracker gives
JITTER = 0.004               # small movement per update, as a fraction of the frame width
MAX_TOTAL = 40               # Shift+. adds people up to this number
LABEL_RED = (200, 30, 30)


def is_fake(face):
    return face.id is not None and face.id >= ID_BASE


class FakeFaces:
    def __init__(self, total=None, flicker=None, seed=None):
        self.on = False
        self.total = config.TEST_FACES if total is None else total
        self.flicker = config.TEST_FLICKER if flicker is None else flicker
        self.hidden = 0
        self.rng = random.Random(seed)

    @property
    def visible(self):
        return self.total - self.hidden

    def hide_one(self):
        self.hidden = min(self.total, self.hidden + 1)

    def show_one(self):
        self.hidden = max(0, self.hidden - 1)

    def add_person(self):
        self.total = min(MAX_TOTAL, self.total + 1)

    def remove_person(self):
        self.total = max(0, self.total - 1)
        self.hidden = min(self.hidden, self.total)

    def boxes(self, w, h):
        """The place of each face (x1, y1, x2, y2) in frame pixels: the front row first (larger
        faces), then the back row. Faces are hidden from the end, so the back row hides first."""
        front = math.ceil(self.total / 2)
        rows = [(front, 0.55, 0.065), (self.total - front, 0.30, 0.045)]   # (faces, center y, width)
        out = []
        for n, cy, fw in rows:
            fw = min(fw, 0.8 * 0.76 / max(1, n))   # many faces: smaller, so they do not overlap
            for i in range(n):
                cx = 0.12 + 0.76 * (i + 0.5) / n
                bw, bh = fw * w, fw * w * 1.25
                out.append((cx * w - bw / 2, cy * h - bh / 2, cx * w + bw / 2, cy * h + bh / 2))
        return out

    @staticmethod
    def body_box(face, h):
        """A body box around a face box: shoulders wide, from just above the head down."""
        x1, y1, x2, y2 = face
        fw, fh = x2 - x1, y2 - y1
        cx = (x1 + x2) / 2
        return (cx - 1.3 * fw, y1 - 0.15 * fh, cx + 1.3 * fw, min(h, y2 + 3.5 * fh))

    def apply(self, det):
        """det with the simulated people added; det itself does not change.
        Every person has a body; only the faces that are not hidden are added."""
        if not self.on:
            return det
        faces, bodies = [], []
        j = JITTER * det.frame_w
        for i, box in enumerate(self.boxes(det.frame_w, det.frame_h)):
            dx, dy = self.rng.uniform(-j, j), self.rng.uniform(-j, j)
            x1, y1, x2, y2 = box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy
            if self.rng.random() >= self.flicker:     # else the AI misses this body in this update
                bodies.append(Obj(id=ID_BASE + i, name="person", conf=0.90,
                                  box=self.body_box((x1, y1, x2, y2), det.frame_h)))
            if i < self.visible and self.rng.random() >= self.flicker:
                faces.append(Face(id=ID_BASE + i, box=(x1, y1, x2, y2), conf=0.90))
        return replace(det, faces=det.faces + faces, persons=det.persons + bodies)

    def draw(self, r, det):
        """A ring on each simulated place (bright: in view, dim: hidden) and a label."""
        for i, box in enumerate(self.boxes(det.frame_w, det.frame_h)):
            rect = r.rect(box)
            color = (230, 230, 230) if i < self.visible else (90, 90, 90)
            pygame.draw.ellipse(r.screen, color, rect, r.u(2))
        msg = (f"TEST MODE: simulated people: {self.total}, hidden faces: {self.hidden}"
               f"   (Y: off   , / . face   Shift: person)")
        pad = r.u(8)
        img = r.font(20, True).render(msg, True, (255, 255, 255))
        box = pygame.Rect(r.u(10), r.top + r.u(10), img.get_width() + 2 * pad, img.get_height() + 2 * pad)
        pygame.draw.rect(r.screen, LABEL_RED, box, border_radius=r.u(6))
        r.screen.blit(img, (box.x + pad, box.y + pad))
