"""Game A - Fool the AI! The team makes the AI see a target object, with anything they have.

A live meter shows how sure the AI is that it sees the target; the best value of the
round is the team score. A real object is allowed too: the AI decides, not us.
"""
import random

import pygame

from .. import config
from ..texts import object_name, t
from .base import CYAN, RoundGame, conf_color
from ..draw import WHITE, YELLOW


class FoolGame(RoundGame):
    game_id = "fool"
    name_key = "fool_name"
    label = "Fool the AI"

    def __init__(self, app=None):
        super().__init__(app)
        self.round_time = config.FOOL_ROUND_TIME
        self.target = random.choice(config.FOOL_TARGETS)
        self.current = self.best = 0.0
        self.target_box = None
        self.now_seen = None          # (english name, conf) of the strongest object right now

    def next_target(self):
        self.target = random.choice([x for x in config.FOOL_TARGETS if x != self.target])

    def extra_key(self, key, app):
        if key == pygame.K_t and self.phase in ("wait", "result"):
            self.next_target()
            return True
        return False

    def start_round(self, now):
        self.current = self.best = 0.0

    def observe(self, app, det, now, dt):
        hits = [o for o in det.all_objects if o.name == self.target]
        top = max(hits, key=lambda o: o.conf, default=None)
        self.current = top.conf if top else 0.0
        self.target_box = top.box if top else None
        seen = [o for o in det.all_objects if o.conf >= config.FOOL_SEEN_MIN]
        best_now = max(seen, key=lambda o: o.conf, default=None)
        self.now_seen = (best_now.name, best_now.conf) if best_now else None

    def play(self, app, det, now, dt):
        self.best = max(self.best, self.current)

    def score(self):
        return round(self.best, 3), self.target

    def result_lines(self):
        target = object_name(self.result["detail"])
        return (t("team_score", p=f"{self.result['score']:.0%}", target=target),
                t("fool_zero", target=target))

    def draw(self, r, app, det, now):
        sw, sh = r.size
        for o in det.objects:
            if o.name != self.target:
                r.labeled_box(o.box, f"{object_name(o.name)} {o.conf:.0%}", (120, 120, 120), width=2, size=20)
        if self.target_box is not None and self.phase != "result":
            r.labeled_box(self.target_box, f"{object_name(self.target)} {self.current:.0%}",
                          conf_color(self.current), width=5, size=26)
        self.draw_task(r, t("fool_task", target=object_name(self.target).upper()))
        if self.draw_phase(r, app, now):
            return

        # live meter at the bottom
        meter = pygame.Rect(r.u(60), sh - r.u(110), sw - r.u(120), r.u(40))
        r.shade(pygame.Rect(0, sh - r.u(190), sw, r.u(190)), 170)
        r.text(t("ai_thinks", p=f"{self.current:.0%}", target=object_name(self.target)), 30, WHITE,
               midleft=(meter.x, sh - r.u(150)))
        seen = t("ai_sees_nothing") if not self.now_seen else \
            t("ai_sees_now", name=object_name(self.now_seen[0]), p=f"{self.now_seen[1]:.0%}")
        r.text(seen, 24, CYAN, midright=(meter.right, sh - r.u(150)))
        r.bar(meter, self.current, conf_color(self.current))
        if self.phase == "playing" and self.best > 0:
            bx = meter.x + int(meter.w * self.best)
            pygame.draw.line(r.screen, YELLOW, (bx, meter.y - r.u(10)), (bx, meter.bottom + r.u(10)), r.u(4))
            r.text(t("best", p=f"{self.best:.0%}"), 18, YELLOW, shadow=False, midtop=(bx, meter.bottom + r.u(10)))
        if self.phase == "wait":
            r.text(t("fool_hint"), 24, WHITE, shadow=False, midtop=(sw // 2, sh - r.u(60)))
        self.draw_instructions(r)
