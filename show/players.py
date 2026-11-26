"""Players: people close to the screen, followed by their body so they stay known when the face disappears.

The modern face model (YOLO) decides who joins. A face belongs to the body whose head area
contains it (face_owner). In stage 3 the active face detector decides
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


def face_owner(face, bodies):
    """The id of the body this face belongs to, or None. A face belongs to a body when its center
    lies in the head area of the body box (the top config.HEAD_PART). With several such bodies,
    the body whose center line is closest to the face wins (in body widths). So when two people
    overlap, a face stays with its own body; a smaller box in front does not take it."""
    cx, cy = (face.box[0] + face.box[2]) / 2, (face.box[1] + face.box[3]) / 2
    best, best_cost = None, None
    for b in bodies.values():
        x1, y1, x2, y2 = b.box
        w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
        if not (x1 <= cx <= x2 and y1 <= cy <= y1 + config.HEAD_PART * h):
            continue
        cost = abs(cx - (x1 + x2) / 2) / w
        if best is None or cost < best_cost:
            best, best_cost = b.id, cost
    return best


def match_to_bodies(faces, bodies, strength):
    """{body id: strongest face that belongs to that body} (see face_owner)."""
    found = {}
    for f in faces:
        best = face_owner(f, bodies)
        if best is not None and (best not in found or strength(f) > strength(found[best])):
            found[best] = f
    return found


def why_not_player(det, bodies, body_face, players):
    """For the operator (debug): [(box, reason)] for each person in view who is not a player."""
    out = []
    for o in det.persons:
        if o.id is None:
            out.append((o.box, "no tracker ID yet"))
    lost = {}   # body id -> a clear face in its head area that went to another body
    for f in det.faces:
        if f.conf < config.FACE_VISIBLE_CONF:
            continue
        owner = face_owner(f, bodies)
        for bid, b in bodies.items():
            if bid != owner and bid not in body_face and face_owner(f, {bid: b}) == bid:
                lost[bid] = f
    full = len(players) >= config.MAX_PLAYERS
    for bid, b in bodies.items():
        if bid in players:
            continue
        f = body_face.get(bid)
        if f is None:
            out.append((b.box, "face went to another body" if bid in lost else "no face in head area"))
        elif not is_big(f, det.frame_w):
            out.append((b.box, f"face too small ({f.width / det.frame_w:.0%} < {config.PLAYER_MIN_FACE:.0%})"))
        elif f.conf < config.FACE_VISIBLE_CONF:
            out.append((b.box, f"face score low ({f.conf:.0%} < {config.FACE_VISIBLE_CONF:.0%})"))
        elif full:
            out.append((b.box, f"game full ({len(players)}/{config.MAX_PLAYERS})"))
    used = {id(f) for f in body_face.values()}
    for f in det.faces:
        if id(f) not in used and f.conf >= config.FACE_VISIBLE_CONF and is_big(f, det.frame_w)                 and face_owner(f, bodies) is None:
            out.append((f.box, "face without a body box"))
    return out


class PlayerTracker:
    def __init__(self):
        self.players = {}
        self.why_not = []       # [(box, reason)] for people in view who are not players (debug)

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

        # the limit went down (operator panel): the players with the smallest body leave first
        while len(self.players) > config.MAX_PLAYERS:
            smallest = min(self.players.values(), key=lambda p: box_area(p.body) if p.body else 0.0)
            del self.players[smallest.body_id]
        self.why_not = why_not_player(det, bodies, body_face, self.players)
