"""Attract mode: passers-by see themselves with a colored box and an invitation to come closer.

Every person gets a box, also far away: the person detector finds whole bodies, which works at
a larger distance than the face detector. A person close to the screen gets a box around the
face instead. The color belongs to the person (by tracker id), so it says: the computer sees
you, and knows you are someone else than your neighbor.
"""
from .. import config
from ..leaderboard import draw_board
from ..draw import WHITE, YELLOW
from ..players import box_area, is_big, match_to_bodies
from ..texts import t
from .base import Stage

PERSON_COLORS = [(80, 180, 255), (255, 140, 60), (90, 230, 160), (255, 100, 200),
                 (255, 220, 70), (170, 120, 255), (60, 230, 230), (255, 90, 90),
                 (180, 240, 80), (255, 170, 200), (140, 160, 255), (240, 240, 240)]


class AttractStage(Stage):
    name = "attract mode"

    def __init__(self):
        self.colors = {}   # tracker id -> color; a new person gets a color nobody in view has

    def person_color(self, track_id, visible):
        """The color of this person. Colors of people who left the image become free again."""
        self.colors = {i: c for i, c in self.colors.items() if i in visible}
        if track_id not in self.colors:
            used = set(self.colors.values())
            free = [c for c in PERSON_COLORS if c not in used]
            self.colors[track_id] = free[0] if free else PERSON_COLORS[len(self.colors) % len(PERSON_COLORS)]
        return self.colors[track_id]

    def action_label(self, app):
        return "Start session"

    def draw(self, r, app, det, now):
        self.draw_people(r, det)
        if not app.overlays:
            return
        sw, sh = r.size
        r.banner(sh - r.u(150), r.u(150))
        invites = t("invites")
        invite = invites[int(now / config.INVITE_SWITCH) % len(invites)]
        r.text(invite, 64, WHITE, center=(sw // 2, sh - r.u(95)))
        r.text(t("faces_today", n=app.faces_today.count), 26, YELLOW, center=(sw // 2, sh - r.u(32)))

        # now and then: a leaderboard with a challenge, on the right side
        games = app.board.games_with_entries()
        if games and now % config.ATTRACT_BOARD_EVERY < config.ATTRACT_BOARD_TIME:
            game = games[int(now // config.ATTRACT_BOARD_EVERY) % len(games)]
            best = app.board.best(game)
            draw_board(r, app.board, game, subtitle=t("beat", team=best["team"]),
                       topright=(sw - r.u(30), r.u(30)))

    def draw_people(self, r, det):
        """A box per person, the largest first, at most config.ATTRACT_MAX_BOXES:
        the face box when the face is close and clearly seen, else the body box."""
        bodies = {b.id: b for b in det.persons if b.id is not None}
        faces = match_to_bodies(det.faces, bodies, lambda f: f.conf)
        used = {id(f) for f in faces.values()}
        items = []   # (size, box, label, key for the color)
        for bid, b in bodies.items():
            f = faces.get(bid)
            if f is not None and f.conf >= config.FACE_VISIBLE_CONF and is_big(f, det.frame_w):
                items.append((box_area(b.box), f.box, _face_label(f), ("body", bid)))
            else:
                items.append((box_area(b.box), b.box, t("person"), ("body", bid)))
        for f in det.faces:  # a close face without a body box (for example only the head in view)
            if id(f) not in used and f.conf >= config.FACE_VISIBLE_CONF and is_big(f, det.frame_w):
                items.append((box_area(f.box) * 4, f.box, _face_label(f), ("face", f.id)))
        items.sort(key=lambda it: -it[0])
        items = items[:config.ATTRACT_MAX_BOXES]
        visible = {key for *_, key in items}
        for _, box, label, key in items:
            r.labeled_box(box, label, self.person_color(key, visible), width=4, size=22)


def _face_label(f):
    return f"{t('face')} {f.conf:.0%}" if config.SHOW_SCORES else t("face")
