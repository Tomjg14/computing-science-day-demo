"""The face that all explanation pages follow.

It is picked once: the largest face that is clearly seen (the closest one, so the pages have
enough detail). After that it never changes by itself, also when the face is covered or lost:
the pages then keep its last place, so the effect of covering stays visible. TAB picks the next
face (from left to right). The lock is dropped when a new session starts.

The face tracker can give the same face a new id after a short loss, so the lock also follows a
face that appears close to its last place (within config.FACE_LOCK_MATCH face widths).
One missed update does not count: the score stays for config.FACE_LOCK_HOLD seconds, so one
flicker of the model does not make it jump to "not found".
Only the box, the id and the score are kept: no image.
"""
from . import config


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


class FaceLock:
    def __init__(self):
        self.reset()

    def reset(self):
        self.id = self.box = None
        self.conf = None           # score at the locked place now; None = not found
        self.seen_at = None

    @property
    def locked(self):
        return self.box is not None

    def _match(self, faces):
        if self.id is not None:
            same = [f for f in faces if f.id == self.id]
            if same:
                return same[0]
        cx, cy = _center(self.box)
        limit = config.FACE_LOCK_MATCH * (self.box[2] - self.box[0])
        near = [f for f in faces if (_center(f.box)[0] - cx) ** 2 + (_center(f.box)[1] - cy) ** 2 <= limit ** 2]
        return min(near, key=lambda f: (_center(f.box)[0] - cx) ** 2 + (_center(f.box)[1] - cy) ** 2,
                   default=None)

    def _take(self, f, now):
        self.id, self.box, self.conf, self.seen_at = f.id, f.box, f.conf, now

    def update(self, faces, now):
        """faces: all faces of this update (from config.FACE_DETECT_CONF up)."""
        if not self.locked:
            clear = [f for f in faces if f.conf >= config.FACE_VISIBLE_CONF]
            if clear:
                self._take(max(clear, key=lambda f: f.box[2] - f.box[0]), now)
            return
        f = self._match(faces)
        if f is not None:
            self._take(f, now)
        elif now - self.seen_at > config.FACE_LOCK_HOLD:
            self.conf = None

    def next(self, faces, now):
        """TAB: lock the next clearly seen face to the right (after the last one: the first).
        Returns True when the lock changed."""
        clear = sorted((f for f in faces if f.conf >= config.FACE_VISIBLE_CONF), key=lambda f: _center(f.box)[0])
        if not clear:
            return False
        if not self.locked:
            self._take(clear[0], now)
            return True
        cx = _center(self.box)[0]
        current = self._match(clear)
        others = [f for f in clear if f is not current]
        if not others:
            return False
        right = [f for f in others if _center(f.box)[0] > cx]
        self._take(right[0] if right else others[0], now)
        return True
