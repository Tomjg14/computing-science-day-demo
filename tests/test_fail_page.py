"""Tests for the page "why can the AI miss a face?" (show/stages/fail_page.py)."""
import unittest
from types import SimpleNamespace

import numpy as np

from show.stages.fail_page import FailExplainer, measure

FRAME = np.zeros((480, 640, 3), np.uint8)


def face(x, conf=0.9, w=100):
    return SimpleNamespace(id=x, box=(x, 100, x + w, 100 + w), conf=conf, width=w)


class MeasureTest(unittest.TestCase):
    def test_flat_area_has_no_contrast(self):
        frame = np.full((100, 100, 3), 200, np.uint8)
        m = measure(frame, (10, 10, 60, 60))
        self.assertEqual(m["width"], 50)
        self.assertAlmostEqual(m["light"], 200, delta=1)
        self.assertAlmostEqual(m["contrast"], 0, delta=0.01)

    def test_half_light_half_dark_has_contrast(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        frame[:, 50:] = 200
        m = measure(frame, (0, 0, 100, 100))
        self.assertAlmostEqual(m["light"], 100, delta=1)
        self.assertAlmostEqual(m["contrast"], 100, delta=1)

    def test_box_outside_the_frame_is_cut(self):
        m = measure(np.zeros((100, 100, 3), np.uint8), (-50, -50, 40, 40))
        self.assertEqual(m["width"], 40)
        self.assertIsNone(measure(np.zeros((100, 100, 3), np.uint8), (200, 200, 300, 300)))


class PageTest(unittest.TestCase):
    def test_page_shows_the_locked_face_and_its_score(self):
        from show.facelock import FaceLock
        lock = FaceLock()
        lock.update([face(0, w=60), face(300, w=120)], 0.0)
        page = FailExplainer()
        page.open(0.0)
        page.update(SimpleNamespace(frame=FRAME, face_lock=lock), None, 0.1)
        self.assertEqual((page.box[0], page.score), (300, 0.9))
        self.assertEqual(page.numbers["width"], 120)

    def test_page_without_a_face_waits(self):
        from show.facelock import FaceLock
        page = FailExplainer()
        page.open(0.0)
        page.update(SimpleNamespace(frame=FRAME, face_lock=FaceLock()), None, 0.1)
        self.assertIsNone(page.box)


if __name__ == "__main__":
    unittest.main()
