"""Game D - Confusion meter. The team makes the AI change its mind about one object as often
as possible: turn it, cover part of it, tilt it.

The game follows the largest object. A change counts when the new label stays for a few
detection updates in a row, so flicker gives no free points.
"""
import pygame

from .. import config
from ..draw import WHITE, YELLOW
from ..texts import object_name, t
from .base import CYAN, RoundGame, conf_color

CHAIN_SHOWN = 6
POP_TIME = 0.6


class ConfuseGame(RoundGame):
    game_id = "confuse"
    name_key = "confuse_name"
    label = "Confusion meter"

    def __init__(self, app=None):
        super().__init__(app)
        self.round_time = config.CONFUSE_ROUND_TIME
        self.main = None          # the largest object right now
        self.start_round(0.0)

    def start_round(self, now):
        self.label_now = None     # the stable label of the main object
        self.candidate, self.candidate_n = None, 0
        self.chain = []           # [(english label, time it appeared)]
        self.changes = 0

    def observe(self, app, det, now, dt):
        objs = [o for o in det.all_objects if o.conf >= config.CONFUSE_MIN_CONF]
        self.main = max(objs, key=lambda o: (o.box[2] - o.box[0]) * (o.box[3] - o.box[1]), default=None)

    def play(self, app, det, now, dt):
        if self.main is None:
            return  # the object left the image: no point, no reset
        name = self.main.name
        if name == self.candidate:
            self.candidate_n += 1
        else:
            self.candidate, self.candidate_n = name, 1
        if self.candidate_n >= config.CONFUSE_STABLE and name != self.label_now:
            if self.label_now is not None:
                self.changes += 1
            self.label_now = name
            self.chain.append((name, now))

    def score(self):
        return self.changes, " > ".join(n for n, _ in self.chain)

    def result_lines(self):
        return t("confuse_score", n=self.result["score"]), t("confuse_zero")

    def draw(self, r, app, det, now):
        sw, sh = r.size
        if self.main is not None and self.phase != "result":
            r.labeled_box(self.main.box, f"{object_name(self.main.name)} {self.main.conf:.0%}",
                          conf_color(self.main.conf), width=5, size=26)
        self.draw_task(r, t("confuse_task"))
        if self.draw_phase(r, app, now):
            return

        # bottom: the chain of labels and the count
        r.shade(pygame.Rect(0, sh - r.u(150), sw, r.u(150)), 170)
        if self.phase == "wait":
            r.text(t("confuse_hint"), 30, WHITE, center=(sw // 2, sh - r.u(75)))
            self.draw_instructions(r)
            return
        r.text(t("confuse_count", n=self.changes), 34, YELLOW, midleft=(r.u(40), sh - r.u(115)))
        x, y = r.u(40), sh - r.u(55)
        shown = self.chain[-CHAIN_SHOWN:]
        if len(self.chain) > CHAIN_SHOWN:
            x = r.text("...  ", 30, WHITE, midleft=(x, y)).right
        for i, (name, at) in enumerate(shown):
            newest = i == len(shown) - 1
            pop = max(0.0, 1 - (now - at) / POP_TIME) if newest else 0.0
            size = int(30 + 18 * pop)
            x = r.text(object_name(name), size, CYAN if newest else WHITE, midleft=(x, y)).right
            if not newest:
                x = r.text("  >  ", 30, YELLOW, midleft=(x, y)).right
        self.draw_instructions(r)
