"""The text steps of the explanation pages: the presenter shows each step, nothing moves on by
itself."""
import unittest

from show.stages.page_parts import Steps


class StepsTest(unittest.TestCase):
    def setUp(self):
        self.s = Steps()
        self.s.total = 3          # a page with 3 steps has drawn itself

    def test_a_page_starts_with_no_steps(self):
        self.assertEqual(self.s.shown, 0)
        self.assertEqual(self.s.newest, -1)

    def test_next_shows_one_step_at_a_time_up_to_the_last(self):
        for want in (1, 2, 3):
            self.assertTrue(self.s.next(now=10.0 + want))
            self.assertEqual(self.s.shown, want)
        self.assertFalse(self.s.next(now=20.0))
        self.assertEqual(self.s.shown, 3)

    def test_back_removes_the_newest_step(self):
        self.s.next(1.0)
        self.s.next(2.0)
        self.assertTrue(self.s.back())
        self.assertEqual(self.s.newest, 0)
        self.s.back()
        self.assertFalse(self.s.back())
        self.assertEqual(self.s.shown, 0)

    def test_age_counts_from_when_the_step_appeared(self):
        self.s.next(5.0)
        self.s.next(9.0)
        self.assertAlmostEqual(self.s.age(0, 12.0), 7.0)
        self.assertAlmostEqual(self.s.age(1, 12.0), 3.0)
        self.assertEqual(self.s.age(2, 12.0), 0.0)     # not on screen yet

    def test_time_alone_does_not_add_a_step(self):
        self.s.next(1.0)
        self.s.age(0, 1000.0)
        self.assertEqual(self.s.shown, 1)

    def test_reset_starts_again_and_needs_a_new_draw(self):
        self.s.next(1.0)
        self.s.reset()
        self.assertEqual(self.s.shown, 0)
        self.assertFalse(self.s.next(2.0))   # total is unknown until the new page draws its steps


if __name__ == "__main__":
    unittest.main()
