"""Game D - Confusion meter. The team makes the AI change its mind about one object as often
as possible: turn it, cover part of it, tilt it.

Before a round, one object gets a box: that is the object that will count. The game prefers
an object held in front of a person (its center inside a person box, and smaller than that
person), then the largest one; it never picks furniture (config.CONFUSE_SKIP). T moves the box
to the next object. SPACE locks it. During the round the game follows only that object, by its tracker id (which stays
when the label changes) or, when the id is lost, by box overlap with its last position. So
objects in the background give no points. A change counts when the new label stays for a few
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
        self.main = None          # before a round: the proposed object; in a round: the locked object
        self.pick_id = None       # tracker id the operator chose with T, or None = automatic choice
        self.cands = []           # candidate objects in this update, best first
        self.start_round(0.0)

    def start_round(self, now):
        self.label_now = None     # the stable label of the main object
        self.candidate, self.candidate_n = None, 0
        self.chain = []           # [(english label, time it appeared)]
        self.changes = 0
        # lock the object that has the box now; nothing in view = lock the first one that comes
        self.lock_id = self.main.id if self.main is not None else None
        self.lock_box = self.main.box if self.main is not None else None

    def action_label(self):
        if self.phase in ("wait", "result") and self.main is not None:
            return f"Lock '{self.main.name}' + start"
        return super().action_label()

    def extra_key(self, key, app):
        if key == pygame.K_t and self.phase in ("wait", "result"):
            self._next_pick()
            return True
        return False

    def _next_pick(self):
        """Move the box to the next object in the candidate order (wraps around)."""
        cands = [o for o in self.cands if o.id is not None]
        if not cands:
            return
        ids = [o.id for o in cands]
        cur = self.main.id if self.main is not None else None
        self.pick_id = ids[(ids.index(cur) + 1) % len(ids)] if cur in ids else ids[0]

    def observe(self, app, det, now, dt):
        self.cands = candidates(det)
        if self.phase in ("countdown", "playing"):
            self.main = self._follow(det.all_objects)
        else:
            chosen = next((o for o in self.cands if self.pick_id is not None and o.id == self.pick_id), None)
            if chosen is None:
                self.pick_id = None  # the chosen object left: back to the automatic choice
            self.main = chosen or (self.cands[0] if self.cands else None)

    def _follow(self, objs):
        """The locked object in this update, or None. Weak guesses keep the lock up to date,
        but only labels from CONFUSE_MIN_CONF count."""
        if self.lock_box is None:
            found = self.cands[0] if self.cands else None
        else:
            found = next((o for o in objs if o.id is not None and o.id == self.lock_id), None)
            if found is None:  # the tracker lost the id: take the object at the same place
                objs = [o for o in objs if o.name not in config.CONFUSE_SKIP]
                best = max(objs, key=lambda o: _iou(o.box, self.lock_box), default=None)
                if best is not None and _iou(best.box, self.lock_box) >= config.CONFUSE_LOCK_IOU:
                    found = best
        if found is None:
            return None
        self.lock_id, self.lock_box = found.id, found.box
        return found if found.conf >= config.CONFUSE_MIN_CONF else None

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
        if self.main is not None and self.phase == "wait":
            r.labeled_box(self.main.box, f"{t('confuse_lock')}: {object_name(self.main.name)} {self.main.conf:.0%}",
                          CYAN, width=5, size=26)
        elif self.main is not None and self.phase != "result":
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


def candidates(det):
    """Objects that can be the game object, best first: held in front of a person, then larger.
    Furniture and weak guesses are left out."""
    persons = [p.box for p in det.persons]

    def held(o):
        cx, cy = (o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2
        return any(x1 <= cx <= x2 and y1 <= cy <= y2 and _area(o.box) < _area((x1, y1, x2, y2))
                   for x1, y1, x2, y2 in persons)

    objs = [o for o in det.all_objects if o.conf >= config.CONFUSE_MIN_CONF and o.name not in config.CONFUSE_SKIP]
    return sorted(objs, key=lambda o: (not held(o), -_area(o.box)))


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _iou(a, b):
    """Overlap of two boxes: shared area / combined area (0 = apart, 1 = the same)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0
