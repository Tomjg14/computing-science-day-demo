"""Game D - Confusion meter. The team makes the AI change its mind about one object as often
as possible: turn it, cover part of it, tilt it.

Before a round, the candidate objects get a numbered box; one of them is highlighted: that is
the object that will count. The automatic choice is an object held by a player (players.held_by),
then the largest one; static objects (config.STATIC_OBJECTS) are never candidates. The operator
can choose another one: click it in the panel, or T for the next number. A chosen object stays
chosen, also when its tracker id changes (found again by box overlap). SPACE locks it and
starts. During the round the game follows only that object, by its tracker id (which stays
when the label changes) or, when the id is lost, by box overlap with its last position. So
objects in the background give no points. A change counts when the new label stays for a few
detection updates in a row, so flicker gives no free points.
"""
import pygame

from .. import config
from ..draw import GREY, WHITE, YELLOW
from ..players import held_by, rank_objects
from ..texts import object_name, t
from .base import CYAN, RoundGame, conf_color

CHAIN_SHOWN = 6
POP_TIME = 0.6
PICK_LOST_AFTER = 2.0   # seconds a chosen object may be out of view before the automatic choice returns


class ConfuseGame(RoundGame):
    game_id = "confuse"
    name_key = "confuse_name"
    label = "Confusion meter"

    def __init__(self, app=None):
        super().__init__(app)
        self.round_time = config.CONFUSE_ROUND_TIME
        self.main = None          # before a round: the proposed object; in a round: the locked object
        self.pick_id = None       # tracker id the operator chose, or None = automatic choice
        self.pick_box, self.pick_seen = None, 0.0   # last box and time the chosen object was seen
        self.cands = []           # candidate objects in this update, best first
        self.held_ids = set()     # ids of the candidates that a player holds
        self.numbers = {}         # tracker id -> number on screen; a number stays while the object stays
        self.app = app
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

    def choices(self):
        """[(number, object, held)] of the candidates, by number: for the screen and the panel."""
        return sorted(((self.numbers[o.id], o, o.id in self.held_ids) for o in self.cands), key=lambda c: c[0])

    def pick(self, obj_id):
        """The operator chooses this candidate (panel click)."""
        if self.phase not in ("wait", "result"):
            return
        obj = next((o for o in self.cands if o.id == obj_id), None)
        if obj is not None:
            self.pick_id, self.pick_box, self.pick_seen = obj.id, obj.box, self.app.now if self.app else 0.0
            self.main = obj

    def _next_pick(self):
        """Choose the candidate with the next number (wraps around)."""
        ids = [o.id for _, o, _ in self.choices()]
        if not ids:
            return
        cur = self.main.id if self.main is not None else None
        self.pick(ids[(ids.index(cur) + 1) % len(ids)] if cur in ids else ids[0])

    def observe(self, app, det, now, dt):
        players = list(app.players.players.values())
        objs = [o for o in det.all_objects if o.id is not None]
        self.cands = rank_objects(objs, players, config.CONFUSE_MIN_CONF)[:config.CONFUSE_MAX_CANDIDATES]
        self.held_ids = {o.id for o in self.cands if held_by(o, players) is not None}
        ids = {o.id for o in self.cands}
        self.numbers = {i: n for i, n in self.numbers.items() if i in ids}
        for o in self.cands:  # a new object gets the smallest free number
            if o.id not in self.numbers:
                self.numbers[o.id] = min(set(range(1, len(self.cands) + 2)) - set(self.numbers.values()))
        if self.phase in ("countdown", "playing"):
            self.main = self._follow(det.all_objects)
        else:
            self.main = self._choose(now)

    def _choose(self, now):
        """Before a round: the operator's choice, else the automatic choice."""
        if self.pick_id is not None:
            chosen = next((o for o in self.cands if o.id == self.pick_id), None)
            if chosen is None:  # the tracker may have given it a new id: look at the same place
                best = max(self.cands, key=lambda o: _iou(o.box, self.pick_box), default=None)
                if best is not None and _iou(best.box, self.pick_box) >= config.CONFUSE_LOCK_IOU:
                    chosen = best
            if chosen is not None:
                self.pick_id, self.pick_box, self.pick_seen = chosen.id, chosen.box, now
                return chosen
            if now - self.pick_seen < PICK_LOST_AFTER:
                return None   # out of view for a moment: keep the choice
            self.pick_id = None
            if self.app is not None:
                self.app.tell_operator("Confuse: the chosen object left the image. Automatic choice again.")
        # automatic: keep the current object while it is still a good choice, so the box does not jump
        cur = self.main
        if cur is not None and cur.id in self.numbers and (cur.id in self.held_ids or not self.held_ids):
            return next(o for o in self.cands if o.id == cur.id)
        return self.cands[0] if self.cands else None

    def _follow(self, objs):
        """The locked object in this update, or None. Weak guesses keep the lock up to date,
        but only labels from CONFUSE_MIN_CONF count."""
        if self.lock_box is None:
            found = self.cands[0] if self.cands else None
        else:
            found = next((o for o in objs if o.id is not None and o.id == self.lock_id), None)
            if found is None:  # the tracker lost the id: take the object at the same place
                objs = [o for o in objs if o.name not in config.STATIC_OBJECTS]
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
        if self.phase == "wait":
            for n, o, _ in self.choices():  # numbered boxes; the chosen one is highlighted
                if self.main is not None and o.id == self.main.id:
                    r.labeled_box(o.box, f"{n}  {t('confuse_lock')}: {object_name(o.name)} {o.conf:.0%}",
                                  CYAN, width=5, size=26)
                else:
                    r.labeled_box(o.box, f"{n}  {object_name(o.name)}", GREY, width=2, size=20)
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


def _area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _iou(a, b):
    """Overlap of two boxes: shared area / combined area (0 = apart, 1 = the same)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(0.0, w) * max(0.0, h)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0
