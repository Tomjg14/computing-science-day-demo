"""Tests for the simulated faces of test mode (show/testmode.py)."""
import unittest

from show import config
from show.testmode import ID_BASE, MAX_TOTAL, FakeFaces, is_fake
from show.vision import Detections, Face

W, H = 1280, 720


def real_det():
    return Detections(frame_w=W, frame_h=H, faces=[Face(id=3, box=(0, 0, 50, 50), conf=0.9)])


def fakes(det):
    return [f for f in det.faces if is_fake(f)]


class FakeFacesTest(unittest.TestCase):
    def test_off_changes_nothing(self):
        det = real_det()
        self.assertIs(FakeFaces(total=8, flicker=0).apply(det), det)

    def test_on_adds_all_faces_and_keeps_real_ones(self):
        ff = FakeFaces(total=8, flicker=0)
        ff.on = True
        det = real_det()
        out = ff.apply(det)
        self.assertEqual(len(fakes(out)), 8)
        self.assertEqual([f.id for f in out.faces if not is_fake(f)], [3])
        self.assertEqual(len(det.faces), 1)                  # the original result does not change

    def test_hide_and_show(self):
        ff = FakeFaces(total=8, flicker=0)
        ff.on = True
        for _ in range(3):
            ff.hide_one()
        self.assertEqual(len(fakes(ff.apply(real_det()))), 5)
        ff.show_one()
        self.assertEqual(len(fakes(ff.apply(real_det()))), 6)

    def test_hidden_stays_between_zero_and_total(self):
        ff = FakeFaces(total=4, flicker=0)
        for _ in range(10):
            ff.hide_one()
        self.assertEqual(ff.visible, 0)
        for _ in range(10):
            ff.show_one()
        self.assertEqual(ff.visible, 4)

    def test_ids_are_stable_and_never_mix_with_tracker_ids(self):
        ff = FakeFaces(total=6, flicker=0)
        ff.on = True
        a = [f.id for f in fakes(ff.apply(real_det()))]
        b = [f.id for f in fakes(ff.apply(real_det()))]
        self.assertEqual(a, b)
        self.assertTrue(all(i >= ID_BASE for i in a))

    def test_flicker_misses_some_faces(self):
        ff = FakeFaces(total=10, flicker=0.3, seed=1)
        ff.on = True
        counts = [len(fakes(ff.apply(real_det()))) for _ in range(200)]
        self.assertLess(min(counts), 10)
        self.assertGreater(sum(counts) / len(counts), 5)

    def test_every_person_has_a_body_also_with_a_hidden_face(self):
        ff = FakeFaces(total=8, flicker=0)
        ff.on = True
        for _ in range(3):
            ff.hide_one()
        out = ff.apply(real_det())
        self.assertEqual(len([p for p in out.persons if is_fake(p)]), 8)
        self.assertEqual(len(fakes(out)), 5)

    def test_face_lies_in_the_head_part_of_its_body(self):
        ff = FakeFaces(total=8, flicker=0)
        ff.on = True
        out = ff.apply(real_det())
        bodies = {p.id: p.box for p in out.persons}
        for f in fakes(out):
            bx1, by1, bx2, by2 = bodies[f.id]
            cx, cy = (f.box[0] + f.box[2]) / 2, (f.box[1] + f.box[3]) / 2
            self.assertTrue(bx1 < cx < bx2)
            self.assertTrue(by1 < cy < by1 + config.HEAD_PART * (by2 - by1))
            self.assertLessEqual(by2, H)

    def test_add_and_remove_people(self):
        ff = FakeFaces(total=3, flicker=0)
        ff.on = True
        ff.hide_one()
        ff.hide_one()
        ff.remove_person()
        ff.remove_person()
        self.assertEqual((ff.total, ff.hidden), (1, 1))       # never more hidden faces than people
        for _ in range(MAX_TOTAL + 5):
            ff.add_person()
        self.assertEqual(ff.total, MAX_TOTAL)
        self.assertEqual(len([p for p in ff.apply(real_det()).persons if is_fake(p)]), MAX_TOTAL)

    def test_flicker_misses_some_bodies(self):
        ff = FakeFaces(total=10, flicker=0.3, seed=1)
        ff.on = True
        counts = [len(ff.apply(real_det()).persons) for _ in range(200)]
        self.assertLess(min(counts), 10)

    def test_faces_lie_inside_the_frame_and_do_not_overlap(self):
        boxes = FakeFaces(total=30).boxes(W, H)
        self.assertEqual(len(boxes), 30)
        for x1, y1, x2, y2 in boxes:
            self.assertTrue(0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H)
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                self.assertFalse(a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3])


if __name__ == "__main__":
    unittest.main()
