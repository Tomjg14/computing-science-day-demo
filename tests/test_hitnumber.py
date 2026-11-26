"""Tests for the game logic of Hit the number (show/stages/hitnumber.py). No camera or window."""
import unittest
from types import SimpleNamespace

import pygame

from show import config
from show.stages.hitnumber import HitNumberStage

SEEN = config.FACE_VISIBLE_CONF + 0.1
UNSURE = config.FACE_VISIBLE_CONF - 0.1


def det(n, unsure=0):
    faces = [SimpleNamespace(id=i, box=(i * 60, 0, i * 60 + 50, 50), conf=SEEN) for i in range(n)]
    faces += [SimpleNamespace(id=100 + i, box=(0, 0, 50, 50), conf=UNSURE) for i in range(unsure)]
    return SimpleNamespace(faces=faces)


class Game:
    """The stage with a small stand-in for the app: a clock and a frame label."""

    def __init__(self):
        self.app = SimpleNamespace(now=0.0, frame=None)
        self.stage = HitNumberStage()
        self.stage.enter(self.app)

    def key(self, k):
        return self.stage.on_key(k, self.app)

    def feed(self, counts, step=0.1):
        """One detection update per count; each frame is labelled with its count and index."""
        for i, n in enumerate(counts):
            self.app.now += step
            self.app.frame = (n, i)
            self.stage.update(self.app, det(n), self.app.now, step)

    def measure(self, counts):
        """Feed counts so that the last one comes just after the measuring time ends.
        (n steps of exactly time / n can add up to a bit less than time in floating point.)"""
        self.feed(counts, step=config.HIT_MEASURE_TIME / (len(counts) - 0.5))

    def calibrate(self, counts):
        self.key(pygame.K_SPACE)
        self.measure(counts)

    def round(self, counts):
        self.key(pygame.K_SPACE)                                 # plan -> countdown
        self.feed([0], step=config.HIT_COUNTDOWN_TIME + 0.01)    # countdown -> measure
        self.measure(counts)


class HitNumberTest(unittest.TestCase):
    def test_starts_with_calibration(self):
        self.assertEqual(Game().stage.phase, "calibrate")

    def test_calibration_sets_maximum_and_target(self):
        g = Game()
        g.calibrate([5] * 10)
        self.assertEqual((g.stage.phase, g.stage.maximum, g.stage.target), ("plan", 5, 4))

    def test_one_flicker_does_not_change_the_maximum(self):
        g = Game()
        g.calibrate([5, 5, 4, 5, 5, 9, 5, 5, 5, 5])
        self.assertEqual(g.stage.maximum, 5)

    def test_unsure_faces_do_not_count(self):
        g = Game()
        g.key(pygame.K_SPACE)
        for _ in range(25):
            g.app.now += 0.1
            g.stage.update(g.app, det(3, unsure=2), g.app.now, 0.1)
        self.assertEqual(g.stage.maximum, 3)

    def test_target_stays_between_zero_and_maximum(self):
        g = Game()
        g.calibrate([3] * 10)
        for _ in range(10):
            g.key(pygame.K_UP)
        self.assertEqual(g.stage.target, 3)
        for _ in range(10):
            g.key(pygame.K_DOWN)
        self.assertEqual(g.stage.target, 0)

    def test_round_counts_the_middle_value(self):
        g = Game()
        g.calibrate([6] * 10)
        g.round([2, 2, 2, 3, 2, 2, 6, 2, 2, 2])
        self.assertEqual(g.stage.phase, "reveal")
        self.assertEqual(g.stage.frozen[2], 2)

    def test_even_number_of_counts_takes_the_lower_middle(self):
        g = Game()
        g.calibrate([6] * 10)
        g.round([2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        self.assertEqual(g.stage.frozen[2], 2)

    def test_frozen_image_has_the_counted_number_of_faces(self):
        g = Game()
        g.calibrate([6] * 10)
        g.round([4, 4, 4, 1, 4, 4, 4, 4, 6, 4])
        frame, faces, count = g.stage.frozen
        self.assertEqual(count, 4)
        self.assertEqual(len(faces), 4)
        self.assertEqual(frame[0], 4)                     # an image of a moment with that count
        self.assertEqual(frame[1], 9)                     # the last such moment

    def test_space_during_countdown_cancels(self):
        g = Game()
        g.calibrate([4] * 10)
        g.key(pygame.K_SPACE)
        self.assertEqual(g.stage.phase, "countdown")
        g.key(pygame.K_SPACE)
        self.assertEqual(g.stage.phase, "plan")

    def test_space_after_reveal_drops_the_frozen_image(self):
        g = Game()
        g.calibrate([4] * 10)
        g.round([3] * 10)
        g.key(pygame.K_SPACE)
        self.assertEqual(g.stage.phase, "plan")
        self.assertIsNone(g.stage.frozen)

    def test_backspace_measures_again(self):
        g = Game()
        g.calibrate([4] * 10)
        g.key(pygame.K_BACKSPACE)
        self.assertEqual(g.stage.phase, "calibrate")
        g.calibrate([7] * 10)
        self.assertEqual((g.stage.maximum, g.stage.target), (7, 6))

    def test_backspace_is_ignored_while_measuring(self):
        g = Game()
        g.calibrate([4] * 10)
        g.key(pygame.K_SPACE)
        g.feed([0], step=config.HIT_COUNTDOWN_TIME + 0.01)
        self.assertEqual(g.stage.phase, "measure")
        self.assertFalse(g.key(pygame.K_BACKSPACE))
        self.assertEqual(g.stage.phase, "measure")

    def test_maximum_is_kept_when_the_chapter_opens_again(self):
        g = Game()
        g.calibrate([5] * 10)
        g.stage.enter(g.app)
        self.assertEqual((g.stage.phase, g.stage.maximum), ("plan", 5))


if __name__ == "__main__":
    unittest.main()
