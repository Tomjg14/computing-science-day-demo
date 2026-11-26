"""Hit the number - the group game of the lesson. The AI must count exactly the target number of
faces; the group decides who hides their face.

calibrate -> (SPACE) measure_max -> plan -> (SPACE) countdown -> measure -> reveal -> (SPACE) plan ...

Calibration: everyone looks at the camera, and each face the AI sees gets a dot. SPACE measures
the maximum: faces the camera cannot see do not count, so every target is possible. The target
starts at the maximum minus 1; the operator changes it with UP / DOWN (or the panel) before a
round. A count is the middle value of all counts in config.HIT_MEASURE_TIME seconds, so one
flicker does not decide. The reveal shows the image of a moment with that count, frozen, with a
dot on each counted face. The frozen image lives in memory only and is dropped at the next round
(ADR 0001). BACKSPACE measures the maximum again. No score is kept.

The game counts every face above config.FACE_VISIBLE_CONF, of any size: it does not use players.
"""
import math
from statistics import median_low

import pygame

from .. import config
from ..draw import GREEN, RED, WHITE, YELLOW
from ..texts import t
from .base import Stage

ORANGE = (255, 140, 40)


def seen_faces(det):
    return [f for f in det.faces if f.conf >= config.FACE_VISIBLE_CONF]


class HitNumberStage(Stage):
    name = "hit the number"
    title_key = "l6_title"
    sub_key = "l6_sub"

    def __init__(self):
        self.maximum = None
        self.target = 0

    def enter(self, app):
        self.phase = "calibrate" if self.maximum is None else "plan"
        self.phase_start = app.now
        self.samples = []   # (count, frame, faces) during a measurement
        self.frozen = None  # (frame, faces, count) on the reveal

    def _set(self, phase, now):
        self.phase, self.phase_start = phase, now

    # --- operator ---
    def action_label(self, app):
        return {"calibrate": "Measure the maximum", "plan": "Start the countdown",
                "countdown": "Cancel the countdown", "reveal": "Next round"}.get(self.phase)

    def step_target(self, d):
        top = self.maximum if self.maximum is not None else 99
        self.target = max(0, min(top, self.target + d))

    def recalibrate(self, now):
        self.frozen = None
        self._set("calibrate", now)

    def on_key(self, key, app):
        if key == pygame.K_SPACE:
            if self.phase == "calibrate":
                self.samples = []
                self._set("measure_max", app.now)
            elif self.phase == "plan":
                self._set("countdown", app.now)
            elif self.phase == "countdown":
                self._set("plan", app.now)
            elif self.phase == "reveal":
                self.frozen = None
                self._set("plan", app.now)
            return True
        if key in (pygame.K_UP, pygame.K_DOWN) and self.phase in ("plan", "reveal"):
            self.step_target(1 if key == pygame.K_UP else -1)
            return True
        if key == pygame.K_BACKSPACE and self.phase not in ("measure", "measure_max"):
            self.recalibrate(app.now)
            return True
        return False

    # --- game ---
    def update(self, app, det, now, dt):
        el = now - self.phase_start
        if self.phase in ("measure_max", "measure"):
            faces = seen_faces(det)
            self.samples.append((len(faces), app.frame, faces))
            if el >= config.HIT_MEASURE_TIME:
                self._finish(now)
        elif self.phase == "countdown" and el >= config.HIT_COUNTDOWN_TIME:
            self.samples = []
            self._set("measure", now)

    def _finish(self, now):
        counts = [n for n, _, _ in self.samples] or [0]
        count = median_low(counts)
        if self.phase == "measure_max":
            self.maximum = count
            self.target = max(0, count - 1)
            self.samples = []
            self._set("plan", now)
            return
        # the last moment with exactly this count: its image and faces go on screen
        frame, faces = next(((fr, fs) for n, fr, fs in reversed(self.samples) if n == count), (None, []))
        self.frozen = (frame, faces, count)
        self.samples = []
        self._set("reveal", now)

    # --- screen ---
    def draw(self, r, app, det, now):
        sw, sh = r.size
        el = now - self.phase_start
        if self.phase == "calibrate":
            faces = seen_faces(det)
            for f in faces:
                r.dot((f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2, GREEN)
            self._bottom(r, t("hit_calibrate"), t("hit_calibrate_hint", n=len(faces)))
        elif self.phase == "measure_max":
            for f in seen_faces(det):
                r.dot((f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2, GREEN)
            self._bottom(r, t("hit_measuring"), "")
        elif self.phase == "plan":
            self._target_card(r)
            self._bottom(r, t("hit_plan"), t("hit_plan_hint"))
        elif self.phase == "countdown":
            self._target_card(r)
            n = math.ceil(config.HIT_COUNTDOWN_TIME - el)
            r.text(str(max(1, n)), 200, YELLOW, center=(sw // 2, sh // 2 + r.u(30)))
        elif self.phase == "measure":
            self._target_card(r)
            r.text(t("hit_measuring"), 56, WHITE, center=(sw // 2, sh // 2 + r.u(30)))
        elif self.phase == "reveal" and self.frozen is not None:
            self._draw_reveal(r)

    def _target_card(self, r):
        sw, _ = r.size
        card = pygame.Rect(sw // 2 - r.u(260), r.top + r.u(16), r.u(520), r.u(150))
        r.shade(card, 190, r.u(16))
        r.text(t("hit_target", n=self.target), 64, YELLOW, center=(card.centerx, card.y + r.u(58)))
        r.text(t("hit_max", n=self.maximum), 22, WHITE, center=(card.centerx, card.y + r.u(118)))

    def _bottom(self, r, big, small):
        sw, sh = r.size
        r.banner(sh - r.u(130), r.u(130))
        r.text(big, 40, WHITE, center=(sw // 2, sh - r.u(85)))
        if small:
            r.text(small, 24, YELLOW, center=(sw // 2, sh - r.u(32)))

    def _draw_reveal(self, r):
        frame, faces, count = self.frozen
        if frame is not None:
            r.video(frame)
        for f in faces:
            r.dot((f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2, GREEN)
        diff = count - self.target
        if diff == 0:
            verdict, color = t("hit_exact"), GREEN
        elif diff < 0:
            verdict, color = t("hit_few", n=-diff), ORANGE
        else:
            verdict, color = t("hit_many", n=diff), RED
        sw, sh = r.size
        card = pygame.Rect(sw // 2 - r.u(330), sh - r.u(190), r.u(660), r.u(170))  # below: faces stay free
        r.shade(card, 200, r.u(16))
        r.text(t("hit_counted", n=count, target=str(self.target)), 44, WHITE,
               center=(card.centerx, card.y + r.u(52)))
        r.text(verdict, 52, color, center=(card.centerx, card.y + r.u(122)))
