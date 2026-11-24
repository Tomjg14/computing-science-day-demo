"""Players: people close to the screen, followed by their body so they stay known when the face disappears.

The modern face model (YOLO) decides who joins. In stage 3 the active face detector decides
whether a player's face is seen: YOLO in hard mode, the Haar cascade in easy mode.
"""
from . import config
from .draw import PLAYER_COLORS


class Player:
    def __init__(self, body_id, color):
        self.body_id = body_id
        self.color = color
        self.body = None        # body box, or None when the body is not seen this frame
        self.face = None
        self.haar_hits = 0      # Haar cascade hits on this player's face (easy mode)
        self.last_seen = 0.0
        self.streak = 0.0
        self.won = False        # won at least once this round (counts for "N winners")
        self.won_at = None      # time of the last win, for the short "won" text

    @property
    def face_conf(self):
        return self.face.conf if self.face is not None else 0.0

    @property
    def invisible(self):
        if self.body is None:
            return False
        if config.EVASION_MODE == "easy":
            return self.haar_hits < config.HAAR_MIN_HITS
        return self.face_conf < config.FACE_VISIBLE_CONF


def is_big(face, frame_w):
    return face.width / frame_w >= config.PLAYER_MIN_FACE


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def held_by(obj, players):
    """The player who holds this object, or None. An object counts as held when its center lies
    inside a player's body box and it is much smaller than that body (config.HELD_MAX_SIZE).
    Players are only the people close to the screen, so objects in the background never count."""
    cx, cy = (obj.box[0] + obj.box[2]) / 2, (obj.box[1] + obj.box[3]) / 2
    for p in players:
        if p.body is None:
            continue
        x1, y1, x2, y2 = p.body
        if x1 <= cx <= x2 and y1 <= cy <= y2 and box_area(obj.box) <= config.HELD_MAX_SIZE * box_area(p.body):
            return p
    return None


def rank_objects(objs, players, min_conf=0.0):
    """Objects that a person could hold, best first: held by a player, then larger.
    Static objects (furniture, vehicles) and guesses below min_conf are left out."""
    players = list(players)
    objs = [o for o in objs if o.conf >= min_conf and o.name not in config.STATIC_OBJECTS]
    return sorted(objs, key=lambda o: (held_by(o, players) is None, -box_area(o.box)))


def match_to_bodies(faces, bodies, strength):
    """{body id: strongest face whose center lies in that body}. A face goes to the smallest
    body box that contains its center."""
    found = {}
    for f in faces:
        cx, cy = (f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2
        best, best_area = None, None
        for b in bodies.values():
            x1, y1, x2, y2 = b.box
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                area = (x2 - x1) * (y2 - y1)
                if best is None or area < best_area:
                    best, best_area = b.id, area
        if best is not None and (best not in found or strength(f) > strength(found[best])):
            found[best] = f
    return found


class PlayerTracker:
    def __init__(self):
        self.players = {}

    def reset(self):
        self.players.clear()

    def player_faces(self):
        return {id(p.face) for p in self.players.values() if p.face is not None}

    def update(self, det, now):
        bodies = {b.id: b for b in det.persons if b.id is not None}
        body_face = match_to_bodies(det.faces, bodies, lambda f: f.conf)
        body_haar = match_to_bodies(det.haar_faces or [], bodies, lambda h: h.hits)

        # a body becomes a player when its face is big and clearly seen
        for bid, f in body_face.items():
            if (bid not in self.players and is_big(f, det.frame_w)
                    and f.conf >= config.FACE_VISIBLE_CONF and len(self.players) < config.MAX_PLAYERS):
                used = {p.color for p in self.players.values()}
                color = next((c for c in PLAYER_COLORS if c not in used), PLAYER_COLORS[0])
                self.players[bid] = Player(bid, color)

        for bid, p in list(self.players.items()):
            b = bodies.get(bid)
            if b is not None:
                p.body, p.face, p.last_seen = b.box, body_face.get(bid), now
                p.haar_hits = body_haar[bid].hits if bid in body_haar else 0
            else:
                p.body, p.face, p.haar_hits = None, None, 0
                if now - p.last_seen > config.PLAYER_LOST_AFTER:
                    del self.players[bid]
