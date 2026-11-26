"""An estimate of how many different faces the show saw today, without storing face data.

The face tracker gives a face a new id each time it loses that face for a moment (a turned
head, a person walking behind someone). Counting ids would count one person many times. Two
filters make the estimate better:

1. A new id counts only after it is seen for config.FACE_COUNT_MIN_TIME seconds, so short
   false detections and passers-by do not count.
2. A new id that appears close to where a counted face was lost in the last
   config.FACE_COUNT_MERGE_TIME seconds is taken as the same person and does not count again.
   A new id close to a lost face that does not count yet takes over its timer.

Only track ids, box positions and times are kept, never an image or a face description, so a
person who leaves and comes back later is counted again. That is why the screen says "about".
"""
from . import config

PENDING_GAP = 1.0   # seconds a not-yet-counted face may be out of view and keep its timer


def _center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


class FaceCounter:
    def __init__(self):
        self.count = 0
        self.known = set()        # ids that are counted, or merged with a counted face
        self.pending = {}         # id -> (first time seen, box, last time seen), for ids that do not count yet
        self.last = {}            # known id -> (box, last time seen), for the merge filter

    def update(self, faces, now):
        """faces: the faces of this update that count as clearly seen."""
        seen = {f.id: f for f in faces if f.id is not None}
        for fid, f in seen.items():
            if fid not in self.known:
                self._new_or_pending(fid, f.box, seen, now)
            if fid in self.known:
                self.last[fid] = (f.box, now)
        # forget what is too old: a pending id missed for 1 s starts again; lost faces after the merge time
        self.pending = {i: v for i, v in self.pending.items() if now - v[2] <= PENDING_GAP}
        self.last = {i: v for i, v in self.last.items() if now - v[1] <= config.FACE_COUNT_MERGE_TIME}

    def _new_or_pending(self, fid, box, seen, now):
        if fid not in self.pending:
            lost = self._near(self.last, box, seen, now)
            if lost is not None:              # the same person with a new id: no new count
                del self.last[lost]
                self.known.add(fid)
                return
            lost = self._near({i: (v[1], v[2]) for i, v in self.pending.items()}, box, seen, now)
            if lost is not None:              # a face that does not count yet, with a new id: keep its timer
                self.pending[fid] = self.pending.pop(lost)
        first = self.pending[fid][0] if fid in self.pending else now
        if now - first >= config.FACE_COUNT_MIN_TIME:
            del self.pending[fid]
            self.known.add(fid)
            self.count += 1
        else:
            self.pending[fid] = (first, box, now)

    @staticmethod
    def _near(faces, box, seen, now):
        """The id of a face in faces ({id: (box, last time seen)}) that is not in view now and was
        recently lost close to this box, or None. The caller removes it, so each lost face can
        take over only one new id."""
        cx, cy = _center(box)
        size = box[2] - box[0]
        for fid, (old, at) in faces.items():
            if fid in seen or now - at > config.FACE_COUNT_MERGE_TIME:
                continue
            ox, oy = _center(old)
            limit = config.FACE_COUNT_MERGE_DIST * max(size, old[2] - old[0])
            if (cx - ox) ** 2 + (cy - oy) ** 2 <= limit ** 2:
                return fid
        return None
