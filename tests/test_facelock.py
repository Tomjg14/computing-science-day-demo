"""Tests for the face that the explanation pages follow (show/facelock.py)."""
import unittest
from types import SimpleNamespace

from show import config
from show.facelock import FaceLock

SEEN = config.FACE_VISIBLE_CONF + 0.2
WEAK = (config.FACE_DETECT_CONF + config.FACE_VISIBLE_CONF) / 2


def face(fid, x, w=100, conf=SEEN):
    return SimpleNamespace(id=fid, box=(x, 100, x + w, 100 + w), conf=conf)


class FaceLockTest(unittest.TestCase):
    def test_picks_the_largest_clear_face(self):
        lock = FaceLock()
        lock.update([face(1, 0, w=60), face(2, 300, w=120), face(3, 600, w=200, conf=WEAK)], 0.0)
        self.assertEqual(lock.id, 2)

    def test_never_switches_by_itself(self):
        lock = FaceLock()
        lock.update([face(1, 0, w=80)], 0.0)
        lock.update([face(1, 0, w=80), face(2, 500, w=200)], 1.0)   # a larger face comes in
        self.assertEqual(lock.id, 1)

    def test_covered_face_keeps_its_place_and_reads_not_found(self):
        lock = FaceLock()
        lock.update([face(1, 300)], 0.0)
        lock.update([face(2, 0)], 0.1 + config.FACE_LOCK_HOLD)        # the locked face is gone, another is seen
        self.assertIsNone(lock.conf)
        self.assertEqual(lock.box[0], 300)
        lock.update([face(2, 0)], 60.0)                               # also after a long time
        self.assertEqual(lock.box[0], 300)

    def test_one_missed_update_keeps_the_score(self):
        lock = FaceLock()
        lock.update([face(1, 300)], 0.0)
        lock.update([], config.FACE_LOCK_HOLD / 2)
        self.assertEqual(lock.conf, SEEN)

    def test_follows_a_new_id_at_the_same_place(self):
        lock = FaceLock()
        lock.update([face(1, 300)], 0.0)
        lock.update([face(7, 320), face(8, 900)], 0.5)
        self.assertEqual((lock.id, lock.box[0]), (7, 320))

    def test_follows_a_weak_face_at_its_place(self):
        # a half-covered face: below the "seen" limit, but still the same face with its low score
        lock = FaceLock()
        lock.update([face(1, 300)], 0.0)
        lock.update([face(1, 300, conf=WEAK)], 0.1)
        self.assertEqual(lock.conf, WEAK)

    def test_next_goes_to_the_right_and_wraps(self):
        lock = FaceLock()
        faces = [face(1, 0), face(2, 300), face(3, 600)]
        lock.update([faces[1]], 0.0)
        lock.next(faces, 1.0)
        self.assertEqual(lock.id, 3)
        lock.next(faces, 2.0)
        self.assertEqual(lock.id, 1)

    def test_next_with_only_the_locked_face_changes_nothing(self):
        lock = FaceLock()
        lock.update([face(1, 300)], 0.0)
        self.assertFalse(lock.next([face(1, 300)], 1.0))
        self.assertEqual(lock.id, 1)

    def test_no_clear_face_no_lock(self):
        lock = FaceLock()
        lock.update([face(1, 300, conf=WEAK)], 0.0)
        self.assertFalse(lock.locked)


if __name__ == "__main__":
    unittest.main()
