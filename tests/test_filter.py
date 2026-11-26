"""The filter page must show the real calculation of the model: the numpy version of the first
layer (show/explain.py) must give the same numbers as the layer of the face model itself."""
import unittest

import numpy as np

from show import config
from show.explain import FirstLayerFilters, first_layer

MODEL = config.ROOT / config.FACE_MODEL


@unittest.skipUnless(MODEL.exists(), "face model not downloaded")
class FirstLayerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from ultralytics import YOLO
        cls.torch = torch
        cls.layer = YOLO(str(MODEL)).model.model[0].eval()
        cls.filters = FirstLayerFilters()
        cls.filters._load()

    def check(self, h, w, seed):
        img = np.random.default_rng(seed).random((h, w, 3))
        with self.torch.no_grad():
            ref = self.layer(self.torch.tensor(img.transpose(2, 0, 1)[None], dtype=self.torch.float32))[0].numpy()
        ours = first_layer(img, self.filters.weights, self.filters.bias)
        self.assertEqual(ours.shape, ref.shape)
        np.testing.assert_allclose(ours, ref, atol=1e-3, rtol=1e-3)

    def test_same_numbers_as_the_model_even_size(self):
        self.check(20, 20, 0)

    def test_same_numbers_as_the_model_odd_size(self):
        self.check(21, 17, 1)

    def test_edge_filters_react_to_edges_not_to_light(self):
        # weights that add up to (almost) 0: a flat light or dark window changes the output much
        # less than an edge between light and dark
        flat_dark, flat_light = np.full((6, 6, 3), 0.1), np.full((6, 6, 3), 0.9)
        edge_v, edge_h = flat_dark.copy(), flat_dark.copy()
        edge_v[:, 3:] = 0.9
        edge_h[3:, :] = 0.9
        edges = self.filters.edge_filters()
        self.assertGreaterEqual(len(edges), 4)
        for i in edges:
            w, b = self.filters.weights[i:i + 1], self.filters.bias[i:i + 1]
            out = lambda img: first_layer(img, w, b)[0, 1:, 1:]   # noqa: E731  (inner part: no padding)
            light_change = np.abs(out(flat_light) - out(flat_dark)).max()
            edge_change = max(np.abs(out(edge_v) - out(flat_dark)).max(), np.abs(out(edge_h) - out(flat_dark)).max())
            self.assertLess(light_change, 0.1 * edge_change, f"filter {i}")


if __name__ == "__main__":
    unittest.main()
