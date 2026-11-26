"""Tests for the "about N faces today" estimate (show/facecount.py)."""
import unittest
from types import SimpleNamespace

from show import config
from show.facecount import PENDING_GAP, FaceCounter

MIN = config.FACE_COUNT_MIN_TIME
MERGE = config.FACE_COUNT_MERGE_TIME


def face(fid, x=100, y=100, w=50):
    return SimpleNamespace(id=fid, box=(x, y, x + w, y + w), conf=0.9)


def run(counter, frames, start=0.0, step=0.1):
    """Feed one list of faces per update, step seconds apart; returns the time after the last one."""
    now = start
    for faces in frames:
        counter.update(faces, now)
        now += step
    return now


def seen_for(seconds, *faces, step=0.1):
    return [list(faces)] * (int(seconds / step) + 1)


class FaceCounterTest(unittest.TestCase):
    def test_short_face_does_not_count(self):
        c = FaceCounter()
        run(c, seen_for(MIN * 0.5, face(1)))
        self.assertEqual(c.count, 0)

    def test_face_counts_once_after_min_time(self):
        c = FaceCounter()
        run(c, seen_for(MIN * 3, face(1)))
        self.assertEqual(c.count, 1)

    def test_two_faces_count_twice(self):
        c = FaceCounter()
        run(c, seen_for(MIN + 0.5, face(1, x=0), face(2, x=500)))
        self.assertEqual(c.count, 2)

    def test_new_id_at_same_place_is_same_person(self):
        c = FaceCounter()
        t = run(c, seen_for(MIN + 0.5, face(1)))
        t = run(c, [[]] * 5, start=t)                      # lost for 0.5 s
        run(c, seen_for(MIN + 0.5, face(2, x=110)), start=t)  # the tracker gives a new id
        self.assertEqual(c.count, 1)

    def test_new_id_far_away_is_new_person(self):
        c = FaceCounter()
        t = run(c, seen_for(MIN + 0.5, face(1)))
        t = run(c, [[]] * 5, start=t)
        run(c, seen_for(MIN + 0.5, face(2, x=1000)), start=t)
        self.assertEqual(c.count, 2)

    def test_new_id_after_merge_time_counts_again(self):
        c = FaceCounter()
        t = run(c, seen_for(MIN + 0.5, face(1)))
        run(c, seen_for(MIN + 0.5, face(2)), start=t + MERGE + 1)
        self.assertEqual(c.count, 2)

    def test_lost_face_takes_over_only_one_new_id(self):
        c = FaceCounter()
        t = run(c, seen_for(MIN + 0.5, face(1)))
        t = run(c, [[]] * 3, start=t)
        run(c, seen_for(MIN + 0.5, face(2, x=100), face(3, x=120)), start=t)
        self.assertEqual(c.count, 2)

    def test_pending_timer_moves_to_new_id(self):
        # id 1 is seen for most of the min time, then the tracker renames it to 2 at the same place:
        # 2 must count as soon as the total time passes, not only after a full new min time
        c = FaceCounter()
        t = run(c, seen_for(MIN * 0.8, face(1)))
        run(c, seen_for(MIN * 0.3, face(2, x=105)), start=t)
        self.assertEqual(c.count, 1)

    def test_pending_face_missed_too_long_starts_again(self):
        c = FaceCounter()
        t = run(c, seen_for(MIN * 0.8, face(1)))
        t = run(c, [[]] * int((PENDING_GAP + 0.5) / 0.1), start=t)
        run(c, seen_for(MIN * 0.3, face(1)), start=t)
        self.assertEqual(c.count, 0)

    def test_face_without_id_is_ignored(self):
        c = FaceCounter()
        run(c, seen_for(MIN * 3, face(None)))
        self.assertEqual(c.count, 0)


if __name__ == "__main__":
    unittest.main()
