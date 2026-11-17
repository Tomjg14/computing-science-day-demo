"""Players: people close to the screen, followed by their body so they stay known when the face disappears."""
from . import config
from .draw import PLAYER_COLORS


class Player:
    def __init__(self, body_id, color):
        self.body_id = body_id
        self.color = color
        self.body = None        # body box, or None when the body is not seen this frame
        self.face = None
        self.last_seen = 0.0
        self.streak = 0.0
        self.won = False

    @property
    def face_conf(self):
        return self.face.conf if self.face is not None else 0.0

    @property
    def invisible(self):
        return self.body is not None and self.face_conf < config.FACE_VISIBLE_CONF


def is_big(face, frame_w):
    return face.width / frame_w >= config.PLAYER_MIN_FACE


class PlayerTracker:
    def __init__(self):
        self.players = {}

    def reset(self):
        self.players.clear()

    def player_faces(self):
        return {id(p.face) for p in self.players.values() if p.face is not None}

    def update(self, det, now):
        bodies = {b.id: b for b in det.persons if b.id is not None}

        # match each face to the smallest body box that contains its center
        body_face = {}
        for f in det.faces:
            cx, cy = (f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2
            best, best_area = None, None
            for b in bodies.values():
                x1, y1, x2, y2 = b.box
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    if best is None or area < best_area:
                        best, best_area = b.id, area
            if best is not None and (best not in body_face or f.conf > body_face[best].conf):
                body_face[best] = f

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
            else:
                p.body, p.face = None, None
                if now - p.last_seen > config.PLAYER_LOST_AFTER:
                    del self.players[bid]
