"""The modern-AI page: the face crop sits in the middle of a square, and the page picks the
filters that point at the face (not the ones that are high everywhere)."""
import unittest

import numpy as np

from show.explain import PAD_GRAY, face_contrast, padded_square


class PaddedSquareTest(unittest.TestCase):
    def test_face_in_the_middle_of_a_square(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        img, face = padded_square(frame, (300, 200, 340, 260), margin=0.35)
        self.assertEqual(img.shape[0], img.shape[1])
        x1, y1, x2, y2 = face
        self.assertAlmostEqual((x1 + x2) / 2, 0.5, delta=0.02)
        self.assertAlmostEqual((y1 + y2) / 2, 0.5, delta=0.02)

    def test_outside_the_frame_is_gray_and_the_face_stays_in_the_middle(self):
        frame = np.zeros((480, 640, 3), np.uint8)
        img, face = padded_square(frame, (0, 10, 60, 90), margin=0.35)   # face at the top-left corner
        self.assertEqual(img.shape[0], img.shape[1])
        self.assertTrue((img[0, 0] == PAD_GRAY).all())                  # off the frame: padding
        self.assertAlmostEqual((face[0] + face[2]) / 2, 0.5, delta=0.02)
        self.assertAlmostEqual((face[1] + face[3]) / 2, 0.5, delta=0.02)


class FaceContrastTest(unittest.TestCase):
    def test_a_blob_on_the_face_beats_a_filter_that_is_high_everywhere(self):
        rng = np.random.default_rng(0)
        maps = np.zeros((3, 10, 10))
        maps[0] = 5 + rng.normal(0, 1, (10, 10))      # high everywhere, noisy
        maps[1, 3:7, 3:7] = 2                          # a blob where the face is
        maps[2, 0:2, 0:2] = 3                          # a blob in the corner (background)
        score = face_contrast(maps, (0.25, 0.25, 0.75, 0.75))
        self.assertEqual(int(np.argmax(score)), 1)
        self.assertGreater(score[0], score[2])         # the corner blob is the worst

    def test_no_face_box_inside_falls_back_to_the_mean(self):
        maps = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)
        np.testing.assert_allclose(face_contrast(maps, (0, 0, 1, 1)), maps.mean(axis=(1, 2)))


if __name__ == "__main__":
    unittest.main()
