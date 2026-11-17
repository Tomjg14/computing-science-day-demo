"""Stage 4 - How researchers do it: the simulated attack (Sharif et al., 2016).

The glasses are drawn on the screen only; the model never sees them, and the
"Panda" label is fixed. The screen says so with a SIMULATION tag.
"""
import math

import numpy as np
import pygame
from PIL import Image

from .. import config
from ..draw import RED, WHITE, YELLOW
from ..players import is_big
from ..texts import t
from .base import Stage

MAX_PANDAS = 4


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

        # the story of the paper, on the right
        sw, sh = r.size
        lines = t("s4_story")
        line = r.u(34)
        panel = pygame.Rect(sw - r.u(470), r.u(140), r.u(450), r.u(40) + line * len(lines))
        r.shade(panel, 180, r.u(14))
        for i, s in enumerate(lines):
            color = YELLOW if i >= 5 else WHITE
            r.text(s, 22, color, shadow=False, topleft=(panel.x + r.u(18), panel.y + r.u(20) + i * line))
        r.text("Sharif et al. (2016)", 16, (200, 200, 200), shadow=False, bold=False,
               bottomright=(panel.right - r.u(14), panel.bottom - r.u(8)))
