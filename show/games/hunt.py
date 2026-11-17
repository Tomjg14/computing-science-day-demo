"""Game C - Scavenger hunt. The AI asks for an object; the team shows it to the camera.

An object counts as found when the AI sees it clearly enough for a few updates in a row.
After a while without success the game moves on by itself, so a group is never stuck.
"""
import random

import pygame

from .. import config
from ..draw import WHITE, YELLOW
from ..texts import object_name, t
from .base import GREEN, ORANGE, RoundGame, conf_color

FLASH_TIME = 1.2


class HuntGame(RoundGame):
    game_id = "hunt"
    name_key = "hunt_name"
    label = "Scavenger hunt"

    def __init__(self, app=None):
        super().__init__(app)
        self.round_time = config.HUNT_ROUND_TIME
        self.target = None
        self.current, self.target_box = 0.0, None
        self.start_round(0.0)

    def start_round(self, now):
        self.found = []           # english names of the found objects, in order
        self.target = None
        self.flash = None         # ("found" | "skip", english name, time)

    def _next_target(self, now):
        options = [x for x in config.HUNT_TARGETS if x != self.target]
        self.target, self.target_start, self.streak = random.choice(options), now, 0

    def observe(self, app, det, now, dt):
        if self.target is None:
            self.current, self.target_box = 0.0, None
            return
        hits = [o for o in det.all_objects if o.name == self.target]
        top = max(hits, key=lambda o: o.conf, default=None)
        self.current = top.conf if top else 0.0
        self.target_box = top.box if top else None

    def play(self, app, det, now, dt):
        if self.target is None:
            self._next_target(now)
            return
        self.streak = self.streak + 1 if self.current >= config.HUNT_MIN_CONF else 0
        if self.streak >= config.HUNT_STABLE:
            self.found.append(self.target)
            self.flash = ("found", self.target, now)
            self._next_target(now)
        elif now - self.target_start >= config.HUNT_SKIP_AFTER:
            self.flash = ("skip", self.target, now)
            self._next_target(now)

    def score(self):
        return len(self.found), ", ".join(self.found)

    def result_lines(self):
        return t("hunt_score", n=self.result["score"]), t("hunt_zero")

    def draw(self, r, app, det, now):
        sw, sh = r.size
        playing = self.phase == "playing" and self.target is not None
        if playing and self.target_box is not None:
            r.labeled_box(self.target_box, f"{object_name(self.target)} {self.current:.0%}",
                          conf_color(self.current), width=5, size=26)
        self.draw_task(r, t("hunt_task", target=object_name(self.target).upper()) if playing else t("hunt_wait"))
        if self.draw_phase(r, app, now):
            return

        r.shade(pygame.Rect(0, sh - r.u(150), sw, r.u(150)), 170)
        r.text(t("hunt_count", n=len(self.found)), 34, YELLOW, midleft=(r.u(40), sh - r.u(115)))
        if playing:
            # time left for this object, before the game moves on by itself
            left = max(0.0, 1 - (now - self.target_start) / config.HUNT_SKIP_AFTER)
            bar = pygame.Rect(sw - r.u(340), sh - r.u(128), r.u(300), r.u(24))
            r.bar(bar, left, ORANGE if left < 0.3 else WHITE)
            # the found objects as a row of names
            x = r.u(40)
            for name in self.found[-7:]:
                x = r.text(object_name(name), 26, GREEN, midleft=(x, sh - r.u(55))).right + r.u(24)
        elif self.phase == "wait":
            r.text(t("hunt_hint"), 26, WHITE, center=(sw // 2, sh - r.u(55)))

        if self.flash and now - self.flash[2] < FLASH_TIME:
            kind, name, at = self.flash
            pop = 1 - (now - at) / FLASH_TIME
            if kind == "found":
                r.text(f"{t('hunt_found')} {object_name(name)}", int(56 + 24 * pop), GREEN, center=(sw // 2, sh // 2))
            else:
                r.text(t("hunt_skip"), int(44 + 12 * pop), ORANGE, center=(sw // 2, sh // 2))
        self.draw_instructions(r)
