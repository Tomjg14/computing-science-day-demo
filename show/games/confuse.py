"""Game D - Confusion meter. The team makes the AI change its mind about their objects as often
as possible: turn them, cover part of them, tilt them.

Before a round, the candidate objects get a numbered box; the highlighted ones are the objects
that will count. The automatic choice is every object held by a player (players.held_by); with
no held object, the largest one. Static objects (config.STATIC_OBJECTS) are never candidates.
The operator can choose one object instead: click it in the panel, or T for the next number
(after the last number, T goes back to the automatic choice). A chosen object stays chosen, also
when its tracker id changes (found again by box overlap). SPACE locks the objects and starts.

During the round the game follows each locked object on its own, by its tracker id (which stays
when the label changes) or, when the id is lost, by box overlap with its last position. Each
object has its own chain of labels; the team score is the sum of the changes of all objects. In
the automatic choice, an object that a player picks up during the round joins the game. So
several players can play at the same time, and objects in the background give no points. A
change counts when the new label stays for a few detection updates in a row, so flicker gives
no free points.
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


class Track:
    """One locked object during a round, with its own chain of labels."""

    def __init__(self, obj):
        self.id, self.box = obj.id, obj.box
        self.obj = None           # the object in this update, or None (out of view or too weak)
        self.label_now = None     # the stable label
        self.candidate, self.candidate_n = None, 0
        self.chain = []           # [(english label, time it appeared)]
        self.changes = 0

    def see(self, obj):
        """The object was found in this update: keep its id and place up to date."""
        self.id, self.box = obj.id, obj.box
        self.obj = obj if obj.conf >= config.CONFUSE_MIN_CONF else None

    def step(self, now):
        """Count a change when a new label stays for CONFUSE_STABLE updates. True on a change."""
        if self.obj is None:
            return False  # the object left the image: no point, no reset
        name = self.obj.name
        if name == self.candidate:
            self.candidate_n += 1
        else:
            self.candidate, self.candidate_n = name, 1
        if self.candidate_n < config.CONFUSE_STABLE or name == self.label_now:
            return False
        changed = self.label_now is not None
        self.changes += changed
        self.label_now = name
        self.chain.append((name, now))
        return changed


class ConfuseGame(RoundGame):
    game_id = "confuse"
    name_key = "confuse_name"
    label = "Confusion meter"

    def __init__(self, app=None):
        super().__init__(app)
        self.round_time = config.CONFUSE_ROUND_TIME
        self.counted = []         # before a round: the objects that will count
        self.pick_id = None       # tracker id the operator chose, or None = automatic choice
        self.pick_box, self.pick_seen = None, 0.0   # last box and time the chosen object was seen
        self.cands = []           # candidate objects in this update, best first
        self.held_ids = set()     # ids of the candidates that a player holds
        self.numbers = {}         # tracker id -> number on screen; a number stays while the object stays
        self.app = app
        self.start_round(0.0)

    def start_round(self, now):
        self.tracks = [Track(o) for o in self.counted]   # nothing in view = lock what comes first
        self.shown = None         # the track whose chain shows at the bottom (the last change)

    @property
    def changes(self):
        return sum(tr.changes for tr in self.tracks)

    def action_label(self):
        if self.phase in ("wait", "result") and self.counted:
            n = len(self.counted)
            return f"Lock '{self.counted[0].name}' + start" if n == 1 else f"Lock {n} objects + start"
        return super().action_label()

    def extra_key(self, key, app):
        if key == pygame.K_t and self.phase in ("wait", "result"):
            self._next_pick()
            return True
        return False

    def choices(self):
        """[(number, object, held)] of the candidates, by number: for the screen and the panel."""
        return sorted(((self.numbers[o.id], o, o.id in self.held_ids) for o in self.cands), key=lambda c: c[0])

    def counted_ids(self):
        return {o.id for o in self.counted}

    def pick(self, obj_id):
        """The operator chooses this candidate (panel click). Clicking the choice again goes
        back to the automatic choice."""
        if self.phase not in ("wait", "result"):
            return
        if obj_id == self.pick_id:
            self.pick_id = None
            return
        obj = next((o for o in self.cands if o.id == obj_id), None)
        if obj is not None:
            self.pick_id, self.pick_box, self.pick_seen = obj.id, obj.box, self.app.now if self.app else 0.0
            self.counted = [obj]

    def _next_pick(self):
        """Choose the candidate with the next number; after the last one, the automatic choice."""
        ids = [o.id for _, o, _ in self.choices()]
        if self.pick_id is None:
            if ids:
                self.pick(ids[0])
        elif self.pick_id in ids and ids.index(self.pick_id) + 1 < len(ids):
            self.pick(ids[ids.index(self.pick_id) + 1])
        else:
            self.pick_id = None

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
            self._follow(det.all_objects)
        else:
            self.counted = self._choose(now)

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
                return [chosen]
            if now - self.pick_seen < PICK_LOST_AFTER:
                return []   # out of view for a moment: keep the choice
            self.pick_id = None
            if self.app is not None:
                self.app.tell_operator("Confuse: the chosen object left the image. Automatic choice again.")
        # automatic: every held object
        held = [o for o in self.cands if o.id in self.held_ids]
        if held:
            return held
        # nothing held: the largest object; keep the current one while it is in view, so the box does not jump
        cur = self.counted[0].id if len(self.counted) == 1 else None
        same = next((o for o in self.cands if o.id == cur), None)
        return [same or self.cands[0]] if self.cands else []

    def _follow(self, objs):
        """Find each locked object in this update. Weak guesses keep a lock up to date, but only
        labels from CONFUSE_MIN_CONF count. In the automatic choice, newly held objects join."""
        taken = set()   # python ids of the objects that a track has, so two tracks never share one
        lost = []       # tracks whose tracker id is not in this update
        for tr in self.tracks:
            tr.obj = None
        # 1. by tracker id: this stays when the label changes
        for tr in self.tracks:
            found = next((o for o in objs if o.id is not None and o.id == tr.id and id(o) not in taken), None)
            if found is None:
                lost.append(tr)
            else:
                taken.add(id(found))
                tr.see(found)
        # 2. the tracker lost the id: take the object at the same place
        movable = [o for o in objs if o.name not in config.STATIC_OBJECTS]
        for tr in lost:
            free = [o for o in movable if id(o) not in taken]
            best = max(free, key=lambda o: _iou(o.box, tr.box), default=None)
            if best is not None and _iou(best.box, tr.box) >= config.CONFUSE_LOCK_IOU:
                taken.add(id(best))
                tr.see(best)
        # 3. automatic choice: objects that a player picks up during the round join the game
        if (self.pick_id is not None and self.tracks) or len(self.tracks) >= config.CONFUSE_MAX_CANDIDATES:
            return
        new = [o for o in self.cands if o.id in self.held_ids and id(o) not in taken]
        if not self.tracks and not new and self.cands:
            new = self.cands[:1]   # nothing held and nothing locked: lock the first object that comes
        for o in new[:config.CONFUSE_MAX_CANDIDATES - len(self.tracks)]:
            tr = Track(o)
            tr.see(o)
            self.tracks.append(tr)

    def play(self, app, det, now, dt):
        for tr in self.tracks:
            if tr.step(now) or (self.shown is None and tr.chain):
                self.shown = tr

    def score(self):
        return self.changes, " | ".join(" > ".join(n for n, _ in tr.chain) for tr in self.tracks if tr.chain)

    def result_lines(self):
        return t("confuse_score", n=self.result["score"]), t("confuse_zero")

    def draw(self, r, app, det, now):
        sw, sh = r.size
        if self.phase == "wait":
            counted = self.counted_ids()
            for n, o, _ in self.choices():  # numbered boxes; the objects that count are highlighted
                if o.id in counted:
                    r.labeled_box(o.box, f"{n}  {t('confuse_lock')}: {object_name(o.name)} {o.conf:.0%}",
                                  CYAN, width=5, size=26)
                else:
                    r.labeled_box(o.box, f"{n}  {object_name(o.name)}", GREY, width=2, size=20)
        elif self.phase != "result":
            for tr in self.tracks:
                if tr.obj is None:
                    continue
                box = r.labeled_box(tr.obj.box, f"{object_name(tr.obj.name)} {tr.obj.conf:.0%}",
                                    conf_color(tr.obj.conf), width=5, size=26)
                if tr.changes and now - tr.chain[-1][1] < POP_TIME:   # "+1" above the object that changed
                    pop = 1 - (now - tr.chain[-1][1]) / POP_TIME
                    r.text("+1", int(40 + 24 * pop), YELLOW, midbottom=(box.centerx, box.y - r.u(40)))
        self.draw_task(r, t("confuse_task"))
        if self.draw_phase(r, app, now):
            return

        # bottom: the total count and the chain of labels of the object that changed last
        r.shade(pygame.Rect(0, sh - r.u(150), sw, r.u(150)), 170)
        if self.phase == "wait":
            r.text(t("confuse_hint"), 30, WHITE, center=(sw // 2, sh - r.u(75)))
            self.draw_instructions(r)
            return
        r.text(t("confuse_count", n=self.changes), 34, YELLOW, midleft=(r.u(40), sh - r.u(115)))
        chain = self.shown.chain if self.shown is not None else []
        x, y = r.u(40), sh - r.u(55)
        shown = chain[-CHAIN_SHOWN:]
        if len(chain) > CHAIN_SHOWN:
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
