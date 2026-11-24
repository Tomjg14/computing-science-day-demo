"""Game B - What does the AI think? A silhouette guessing game for the whole audience.

Only an object held by a player plays (players.held_by): the live image shows one '???' box,
on that object. SPACE freezes the image: a segmentation model cuts out that object, which
appears as a black silhouette on a star burst. With no held object, SPACE asks the players
to hold one up and does not freeze; objects in the background never play. The audience guesses; SPACE reveals what the AI
thinks, with the score counting up. No score is kept. The frozen image lives in memory
only and is dropped for the next object (ADR 0001).
"""
import math

import cv2
import numpy as np
import pygame

from .. import config
from ..draw import WHITE, YELLOW
from ..players import held_by, rank_objects
from ..texts import object_name, t
from .base import CYAN, conf_color

RAYS = 20
RAY_A, RAY_B = (70, 30, 120), (110, 50, 180)
STAR_FILL, STAR_EDGE = (255, 210, 60), (255, 140, 40)
COUNT_TIME = 1.4
MATCH_IOU = 0.3   # a cut-out shape is the held object when its box overlaps this much
FLASH_TIME = 0.45


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def star_points(center, outer, inner, spikes, turn=0.0):
    cx, cy = center
    pts = []
    for i in range(spikes * 2):
        rad = outer if i % 2 == 0 else inner
        a = turn + math.pi * i / spikes
        pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    return pts


class GuessGame:
    game_id = "guess"
    name_key = "guess_name"
    label = "What does the AI think?"

    def __init__(self, app):
        self.phase, self.phase_start = "live", 0.0   # live | looking | frozen | reveal
        self.instructions = True
        self.answer = None        # {"name", "conf", "also", "silhouette", "photo"} while frozen
        self.held = None          # the object a player holds now (live phase), or None
        self.none_until = 0.0
        self.none_key = "guess_none"
        self.segmenter = app.segmenter
        self.segmenter.preload()

    # --- operator ---
    def can_name(self):
        return False  # no score in this game

    def action_label(self):
        return {"live": "Freeze: show silhouette", "frozen": "Reveal the answer",
                "reveal": "Next object"}.get(self.phase)

    def on_key(self, key, app):
        if key == pygame.K_i:
            self.instructions = not self.instructions
            return True
        if key != pygame.K_SPACE:
            return False
        if self.phase == "live" and app.frame is not None:
            self.instructions = False
            if self.held is None:  # nothing held: ask for an object, do not freeze
                self.none_until, self.none_key = app.now + 3.0, "guess_hold"
                return True
            frame, target = app.frame.copy(), self.held.box
            self.phase, self.phase_start = "looking", app.now
            self.segmenter.run_async(frame, lambda found: self._on_found(frame, found, target))
        elif self.phase == "frozen":
            self.phase, self.phase_start = "reveal", app.now
        elif self.phase == "reveal":
            self.answer = None
            self.phase = "live"
        return True

    def _on_found(self, frame, found, target):
        """Runs in the segmenter thread: cut out the shape that matches the held object's box."""
        match = max(found, key=lambda f: iou(f["box"], target), default=None)
        if match is None or iou(match["box"], target) < MATCH_IOU:
            self.phase, self.none_until, self.none_key = "live", self.phase_start + 3.0, "guess_none"
            return
        main = match
        also = sorted((f for f in found if f is not main and f["name"] != main["name"]
                       and iou(f["box"], main["box"]) > 0.4), key=lambda f: -f["conf"])
        seen, alternatives = set(), []
        for f in also:
            if f["name"] not in seen:
                seen.add(f["name"])
                alternatives.append((f["name"], f["conf"]))
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = main["box"]
        pad = 0.12 * max(x2 - x1, y2 - y1)
        a, b = int(max(0, x1 - pad)), int(max(0, y1 - pad))
        c, d = int(min(w, x2 + pad)), int(min(h, y2 + pad))
        rgb = cv2.cvtColor(frame[b:d, a:c], cv2.COLOR_BGR2RGB)
        alpha = (main["mask"][b:d, a:c] * 255).astype(np.uint8)
        dark = np.zeros_like(rgb)
        dark[:] = (15, 15, 25)
        self.answer = {"name": main["name"], "conf": main["conf"], "also": alternatives[:3],
                       "silhouette": _rgba(dark, alpha), "photo": _rgba(rgb, alpha)}
        self.phase = "frozen"

    def update(self, app, det, now, dt):
        players = list(app.players.players.values())
        ranked = rank_objects(det.objects, players)
        self.held = ranked[0] if ranked and held_by(ranked[0], players) is not None else None

    # --- screen ---
    def draw(self, r, app, det, now):
        sw, sh = r.size
        if self.phase in ("live", "looking"):
            if self.held is not None and self.phase == "live":  # one box, no label: no spoilers
                r.labeled_box(self.held.box, "???", YELLOW, width=4, size=24)
            r.banner(r.top, r.u(60), 120)
            r.text(t("guess_live"), 36, YELLOW, center=(sw // 2, r.top + r.u(30)))
            if self.phase == "looking":
                r.text(t("guess_looking"), 44, WHITE, center=(sw // 2, sh // 2))
            elif now < self.none_until:
                r.text(t(self.none_key), 36, (255, 140, 40), center=(sw // 2, sh // 2))
            if self.instructions:
                r.instructions(t("instr_title") + ": " + t("guess_name"), t("guess_steps"), t("instr_why"),
                               t("guess_why"), (r.u(24), r.top + r.u(76)))
            return
        if self.answer is None:
            return
        self._draw_stage(r, now)

    def _draw_stage(self, r, now):
        sw, sh = r.size
        el = now - self.phase_start
        r.shade(pygame.Rect(0, 0, sw, sh), 200)
        area = pygame.Rect(r.u(140), r.u(118), sw - r.u(280), sh - r.u(200))
        center = (area.centerx, area.centery + r.u(30))

        # turning rays, clipped to the area
        rays = pygame.Surface(area.size, pygame.SRCALPHA)
        rc = (center[0] - area.x, center[1] - area.y)
        far = max(area.w, area.h)
        turn = now * 0.35
        for i in range(RAYS):
            a0, a1 = turn + 2 * math.pi * i / RAYS, turn + 2 * math.pi * (i + 1) / RAYS
            pts = [rc, (rc[0] + far * math.cos(a0), rc[1] + far * math.sin(a0)),
                   (rc[0] + far * math.cos(a1), rc[1] + far * math.sin(a1))]
            pygame.draw.polygon(rays, RAY_A if i % 2 else RAY_B, pts)
        r.screen.blit(rays, area.topleft)
        pygame.draw.rect(r.screen, YELLOW, area, r.u(4), border_radius=r.u(12))

        # the star burst behind the object
        outer = r.u(225) * (1 + 0.03 * math.sin(now * 4))
        star = star_points(center, outer, outer * 0.74, 14, turn=-now * 0.2)
        pygame.draw.polygon(r.screen, STAR_FILL, star)
        pygame.draw.polygon(r.screen, STAR_EDGE, star, r.u(6))

        # the object: silhouette, then a cross-fade to the real photo
        sil, photo = self.answer["silhouette"], self.answer["photo"]
        scale = r.u(300) / max(sil.get_width(), sil.get_height())
        size = (int(sil.get_width() * scale), int(sil.get_height() * scale))
        pos = (center[0] - size[0] // 2, center[1] - size[1] // 2)
        mix = 0.0 if self.phase == "frozen" else min(1.0, el / FLASH_TIME)
        if mix < 1:
            img = pygame.transform.smoothscale(sil, size)
            img.set_alpha(int(255 * (1 - mix)))
            r.screen.blit(img, pos)
        if mix > 0:
            img = pygame.transform.smoothscale(photo, size)
            img.set_alpha(int(255 * mix))
            r.screen.blit(img, pos)

        # title and answer
        r.text(t("guess_name"), 52, YELLOW, midtop=(sw // 2, area.y + r.u(12)))
        if self.phase == "frozen":
            r.text(t("guess_ask"), 30, WHITE, midbottom=(sw // 2, area.bottom - r.u(14)))
            return
        if el < FLASH_TIME:  # white flash
            flash = pygame.Surface((sw, sh))
            flash.fill(WHITE)
            flash.set_alpha(int(220 * (1 - el / FLASH_TIME)))
            r.screen.blit(flash, (0, 0))
        r.shade(pygame.Rect(area.x + r.u(4), area.bottom - r.u(122), area.w - r.u(8), r.u(118)), 190)
        conf = self.answer["conf"] * min(1.0, el / COUNT_TIME)
        name = object_name(self.answer["name"]).upper()
        r.text(t("guess_reveal", name=name), 52, WHITE, midbottom=(sw // 2, area.bottom - r.u(56)))
        r.text(f"{conf:.0%}", 44, conf_color(conf), midbottom=(sw // 2, area.bottom - r.u(10)))
        if el >= COUNT_TIME and self.answer["also"]:
            items = ", ".join(f"{object_name(n)} {c:.0%}" for n, c in self.answer["also"])
            r.text(t("guess_also", items=items), 24, CYAN, midtop=(sw // 2, area.bottom + r.u(10)))


def _rgba(rgb, alpha):
    data = np.dstack([rgb, alpha]).astype(np.uint8)
    h, w = data.shape[:2]
    return pygame.image.frombuffer(np.ascontiguousarray(data).tobytes(), (w, h), "RGBA").copy()
