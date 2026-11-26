"""Real computations behind the stage 3 explanation: the old AI (Haar cascade) and the new AI
(the layers of the YOLO face model). Nothing here is drawn; the stage draws the results.

Old AI: the cascade file of OpenCV is read and evaluated on one 24 x 24 window, with the same
formula as OpenCV: a test adds and subtracts the pixel sums of 2 or 3 rectangles, divides by
the contrast of the window, and compares the result with its limit. The test then gives the
left or the right number of points. A stage passes when its points reach the stage limit.

New AI: a second copy of the face model (the live one runs in the vision thread) looks at a
crop of the face; hooks copy the output of three layers.
"""
import threading
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from . import config

FEATURE_LAYERS = (0, 4, 8)   # face model layers shown: early (edges), middle (parts), late (whole face)
MAPS_PER_LAYER = 4


class HaarCascadeFile:
    """The tests and stages of an OpenCV Haar cascade file."""

    def __init__(self, path=None):
        root = ET.parse(path or cv2.data.haarcascades + config.HAAR_FILE).getroot()[0]
        self.size = int(root.find("width").text)
        self.features = []   # per feature: [(x, y, w, h, weight)]
        for f in root.find("features"):
            rects = []
            for rect in f.find("rects"):
                x, y, w, h, wt = rect.text.split()
                rects.append((int(x), int(y), int(w), int(h), float(wt)))
            self.features.append(rects)
        self.stages = []     # per stage: (limit, [(feature index, test limit, points left, points right)])
        for s in root.find("stages"):
            tests = []
            for wc in s.find("weakClassifiers"):
                node = wc.find("internalNodes").text.split()
                left, right = (float(v) for v in wc.find("leafValues").text.split())
                tests.append((int(node[2]), float(node[3]), left, right))
            self.stages.append((float(s.find("stageThreshold").text), tests))

    @property
    def n_tests(self):
        return sum(len(tests) for _, tests in self.stages)

    def evaluate(self, window):
        """window: gray image of size x size (uint8). Returns a dict with the result per test of
        the first stage, and how many stages pass in a row (all = a face for the old AI)."""
        win = window.astype(np.float64)
        ii = cv2.integral(win)
        sq = cv2.integral(win * win)

        def rsum(img, x, y, w, h):
            return img[y + h, x + w] - img[y, x + w] - img[y + h, x] + img[y, x]

        # contrast of the window (OpenCV uses the window minus a border of 1 pixel)
        n = (self.size - 2) ** 2
        s, s2 = rsum(ii, 1, 1, self.size - 2, self.size - 2), rsum(sq, 1, 1, self.size - 2, self.size - 2)
        nf = n * s2 - s * s
        nf = np.sqrt(nf) if nf > 0 else 1.0

        stages, passed = [], 0
        for limit, tests in self.stages:
            results, total = [], 0.0
            for fi, tlimit, left, right in tests:
                rects = self.features[fi]
                sums = [rsum(ii, x, y, w, h) for x, y, w, h, _ in rects]
                value = sum(wt * sm for (*_, wt), sm in zip(rects, sums)) / nf
                points = left if value < tlimit else right
                total += points
                results.append({"rects": rects, "sums": sums, "value": value, "limit": tlimit,
                                "points": points, "low": value < tlimit,
                                "points_left": left, "points_right": right})
            ok = total >= limit
            stages.append({"tests": results, "total": total, "limit": limit, "ok": ok})
            if not ok:
                break
            passed += 1
        return {"stages": stages, "passed": passed, "n_stages": len(self.stages), "face": passed == len(self.stages)}


def haar_window(gray, box, haar):
    """The square window the old AI looks at: a Haar cascade hit near the face box when there is
    one (the window the old AI itself finds), else a square around the face box.
    Returns ((x1, y1, x2, y2), found_by_haar)."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h, w = gray.shape[:2]
    a, b = int(max(0, cx - side)), int(max(0, cy - side))
    c, d = int(min(w, cx + side)), int(min(h, cy + side))
    if c - a > 24 and d - b > 24:
        hits = haar.detectMultiScale(gray[b:d, a:c], scaleFactor=1.1, minNeighbors=3,
                                     minSize=(int(side * 0.6), int(side * 0.6)))
        if len(hits):
            hx, hy, hw, hh = max(hits, key=lambda r: r[2] * r[3])
            return (a + hx, b + hy, a + hx + hw, b + hy + hh), True
    half = side / 2
    return (cx - half, cy - half, cx + half, cy + half), False


def cut_window(gray, win, size):
    """The window, cut out and made size x size (as the cascade sees it)."""
    x1, y1, x2, y2 = (int(round(v)) for v in win)
    h, w = gray.shape[:2]
    if x1 < 0 or y1 < 0 or x2 > w or y2 > h or x2 - x1 < 8:
        return None
    return cv2.resize(gray[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


PAD_GRAY = 114   # the gray that the model itself pads its input with


def padded_square(frame, box, margin):
    """The square around a box (like vision.crop_square), with PAD_GRAY where the square goes
    outside the frame, so the face is always in the middle and the square fits the model input
    with no extra padding: each cell of a feature map then covers the same part of this image.
    Returns (image, face box as fractions of the square: x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (0.5 + margin)
    a, b = int(round(cx - half)), int(round(cy - half))
    side = max(2, int(round(2 * half)))
    c, d = a + side, b + side
    h, w = frame.shape[:2]
    img = frame[max(0, b):min(h, d), max(0, a):min(w, c)]
    img = cv2.copyMakeBorder(img, max(0, -b), max(0, d - h), max(0, -a), max(0, c - w),
                             cv2.BORDER_CONSTANT, value=(PAD_GRAY, PAD_GRAY, PAD_GRAY))
    return img, ((x1 - a) / side, (y1 - b) / side, (x2 - a) / side, (y2 - b) / side)


def face_contrast(maps, face):
    """Per channel of a feature map (channels, h, w): how much higher its output is on the face
    than around it, in units of its own spread. High = a filter that points at the face; a filter
    that is high everywhere scores low. face: x1, y1, x2, y2 as fractions of the image."""
    c, mh, mw = maps.shape
    x1, y1, x2, y2 = face
    inside = np.zeros((mh, mw), bool)
    inside[max(0, int(y1 * mh)):int(np.ceil(y2 * mh)), max(0, int(x1 * mw)):int(np.ceil(x2 * mw))] = True
    flat = maps.reshape(c, -1)
    if inside.all() or not inside.any():
        return flat.mean(axis=1)
    m = inside.ravel()
    return (flat[:, m].mean(axis=1) - flat[:, ~m].mean(axis=1)) / (flat.std(axis=1) + 1e-6)


class FaceLayers:
    """A second copy of the face model, for the explanation only. It runs in its own thread,
    one face crop at a time, so the show never waits for it."""

    def __init__(self):
        self.model = None
        self.result = None          # {"crop": rgb, "maps": {layer: array (channels, h, w)}, "conf": float,
        #                              "face": face box as fractions of the crop, or None}
        self.n_filters = None       # number of filters in all layers of the model
        self._busy = False
        self._lock = threading.Lock()
        self._outs = {}

    def _load(self):
        with self._lock:
            if self.model is None:
                from ultralytics import YOLO
                from .vision import _model_path
                self.model = YOLO(_model_path(config.FACE_MODEL, config.FACE_MODEL_URL))
                import torch
                self.n_filters = sum(m.out_channels for m in self.model.model.modules()
                                     if isinstance(m, torch.nn.Conv2d))
                net = self.model.model.model
                for i in FEATURE_LAYERS:
                    net[i].register_forward_hook(lambda mod, inp, out, i=i: self._outs.__setitem__(i, out))

    def preload(self):
        threading.Thread(target=self._load, daemon=True).start()

    def submit(self, crop_bgr, face=None):
        """Run the model on this crop, unless it is still busy with the previous one.
        face: where the face is in the crop (fractions), kept with the result."""
        if self._busy or crop_bgr is None:
            return
        self._busy = True

        def work():
            try:
                self._load()
                r = self.model.predict(crop_bgr, imgsz=320, verbose=False, conf=0.05)[0]
                conf = float(r.boxes.conf.max()) if r.boxes is not None and len(r.boxes) else 0.0
                maps = {i: self._outs[i][0].detach().float().cpu().numpy() for i in FEATURE_LAYERS}
                self.result = {"crop": cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), "maps": maps, "conf": conf,
                               "face": face}
            finally:
                self._busy = False
        threading.Thread(target=work, daemon=True).start()


class ObjectGrid:
    """A second copy of the object model, for the stage 2 explanation. It looks at a smaller copy
    of the camera image (sides a multiple of 32, so there is no padding) and keeps, for every
    square of the grid, the scores of all 80 object types (sigmoid of the model's raw output).
    YOLO uses three grid sizes (squares of 8, 16 and 32 pixels); the coarse grid of 32 pixels
    gets, per square, the highest score of all squares inside it.
    Each square also predicts its own box: 4 distances (left, top, right, bottom) from its center,
    in square sizes. This model (one-to-one head, no extra merging step) gives every final box
    from one square; the focus is that square, found by its score and its box."""

    WIDTH = 512

    def __init__(self):
        self.model = None
        self.result = None   # see _work
        self._busy = False
        self._lock = threading.Lock()
        self._head = {}

    def _load(self):
        with self._lock:
            if self.model is None:
                from ultralytics import YOLO
                from .vision import _model_path
                self.model = YOLO(_model_path(config.OBJECT_MODEL))
                self.model.model.model[-1].register_forward_hook(
                    lambda mod, inp, out: self._head.__setitem__("out", out))

    def preload(self):
        threading.Thread(target=self._load, daemon=True).start()

    def submit(self, frame_bgr, pick):
        """pick(list of (name, conf, box in frame pixels)) -> index of the object to explain, or None."""
        if self._busy or frame_bgr is None:
            return
        self._busy = True
        threading.Thread(target=self._work, args=(frame_bgr, pick), daemon=True).start()

    def _work(self, frame, pick):
        try:
            self._load()
            fh, fw = frame.shape[:2]
            w = self.WIDTH
            h = max(32, int(round(fh * w / fw / 32)) * 32)
            img = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            r = self.model.predict(img, imgsz=w, verbose=False, conf=config.OBJECT_CONF)[0]
            o = self._head["out"][1]["one2one"]
            prob = o["scores"][0].sigmoid().float().cpu().numpy()      # (80, squares of all three grids)
            sizes = [tuple(f.shape[2:]) for f in o["feats"]]            # (rows, cols) per grid, fine to coarse
            gh, gw = sizes[-1]
            heat = np.zeros((gh, gw))
            centers, strides, start = [], [], 0
            for (rh, rw) in sizes:
                n = rh * rw
                best = prob[:, start:start + n].max(axis=0).reshape(rh, rw)
                f = rh // gh
                heat = np.maximum(heat, best.reshape(gh, f, gw, f).max(axis=(1, 3)))
                s = h / rh
                ys, xs = np.mgrid[0:rh, 0:rw]
                centers.append(np.stack([(xs.ravel() + 0.5) * s, (ys.ravel() + 0.5) * s], axis=1))
                strides.append(np.full(n, s))
                start += n
            centers, strides = np.concatenate(centers), np.concatenate(strides)
            dist = o["boxes"][0].float().cpu().numpy() * strides               # (4, squares): l, t, r, b
            own = np.stack([centers[:, 0] - dist[0], centers[:, 1] - dist[1],
                            centers[:, 0] + dist[2], centers[:, 1] + dist[3]], axis=1)   # each square's box
            sx, sy = fw / w, fh / h
            dets = [(self.model.names[int(c)], float(p), (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy))
                    for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
            focus = None
            i = pick(dets) if dets else None
            if i is not None:
                name, conf, (x1, y1, x2, y2) = dets[i]
                cls = next(k for k, v in self.model.names.items() if v == name)
                scale = np.array([sx, sy, sx, sy])
                # the square that gave this box: the same score and the same box
                miss = np.abs(prob[cls] - conf) * 100 + np.abs(own * scale - np.array([x1, y1, x2, y2])).sum(axis=1) / fw
                a = int(np.argmin(miss))
                if miss[a] < 0.05:
                    s = strides[a] / 2
                    top = np.argsort(-prob[:, a])[:5]
                    focus = {"det": i, "center": (centers[a, 0] * sx, centers[a, 1] * sy),
                             "square": tuple((np.array([centers[a, 0] - s, centers[a, 1] - s,
                                                        centers[a, 0] + s, centers[a, 1] + s]) * scale).tolist()),
                             "box": tuple((own[a] * scale).tolist()),
                             "top": [(self.model.names[int(k)], float(prob[k, a])) for k in top]}
            self.result = {"image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "heat": heat,
                           "frame_size": (fw, fh), "dets": dets, "focus": focus}
        finally:
            self._busy = False


def first_layer(x01, weights, bias):
    """The first layer of the YOLO model, in plain numpy: a 3 x 3 window that jumps 2 pixels,
    zero padding of 1 pixel, the sum of weight x pixel over the window and the 3 colors, plus the
    fixed number of the filter, then SiLU (negative results become close to 0).
    x01: (h, w, 3) RGB image with values 0-1. weights: (filters, 3, 3, 3) as (filter, color, row,
    col). bias: (filters,). Returns (filters, ceil(h / 2), ceil(w / 2))."""
    h, w = x01.shape[:2]
    p = np.pad(x01.astype(np.float64), ((1, 1), (1, 1), (0, 0)))
    oh, ow = (h + 1) // 2, (w + 1) // 2
    out = np.zeros((len(weights), oh, ow))
    for dy in range(3):
        for dx in range(3):
            patch = p[dy:dy + 2 * oh:2, dx:dx + 2 * ow:2, :]           # (oh, ow, 3)
            out += np.einsum("hwc,fc->fhw", patch, weights[:, :, dy, dx])
    out += bias[:, None, None]
    return silu(out)


def silu(v):
    return v / (1 + np.exp(-v))


class FirstLayerFilters:
    """The 32 filters of the first layer of the face model: the weights and the fixed number of
    each filter, with the batch normalization folded in, as the model computes them. Loaded once,
    in the background."""

    def __init__(self):
        self.weights = self.bias = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self.weights is not None:
                return
            from ultralytics import YOLO
            from .vision import _model_path
            layer = YOLO(_model_path(config.FACE_MODEL, config.FACE_MODEL_URL)).model.model[0]
            w = layer.conv.weight.detach().float().cpu().numpy().astype(np.float64)
            if hasattr(layer, "bn"):
                bn = layer.bn
                scale = (bn.weight / (bn.running_var + bn.eps).sqrt()).detach().cpu().numpy()
                self.bias = (bn.bias.detach().cpu().numpy() - bn.running_mean.detach().cpu().numpy() * scale)
                self.weights = w * scale[:, None, None, None]
            else:   # already folded
                self.weights = w
                self.bias = layer.conv.bias.detach().cpu().numpy().astype(np.float64)

    def preload(self):
        threading.Thread(target=self._load, daemon=True).start()

    @property
    def ready(self):
        return self.weights is not None

    def edge_filters(self, limit=0.05):
        """Filters that react to edges and hardly to light: measured on flat gray windows (dark and
        light) and on a light-dark edge (across and along). A filter counts when the change from
        dark to light moves its output at most limit x as much as the edge does."""
        dark, light = np.full((6, 6, 3), 0.1), np.full((6, 6, 3), 0.9)
        across, along = dark.copy(), dark.copy()
        across[:, 3:] = 0.9
        along[3:, :] = 0.9
        out = []
        for i in range(len(self.weights)):
            w, b = self.weights[i:i + 1], self.bias[i:i + 1]
            o = [first_layer(img, w, b)[0, 1:, 1:] for img in (dark, light, across, along)]   # inner part
            light_change = np.abs(o[1] - o[0]).max()
            edge_change = max(np.abs(o[2] - o[0]).max(), np.abs(o[3] - o[0]).max())
            if light_change <= limit * edge_change:
                out.append(i)
        return out


_FIRST_LAYER = None


def shared_first_layer():
    """One FirstLayerFilters for all pages (it loads the model once)."""
    global _FIRST_LAYER
    if _FIRST_LAYER is None:
        _FIRST_LAYER = FirstLayerFilters()
    return _FIRST_LAYER


def best_edge_filter(filters, maps):
    """Of the filters that react to edges, the one with the most contrast in these outputs
    (maps: (filters, h, w) of the first layer)."""
    return max(filters.edge_filters(), key=lambda i: float(maps[i].std()))
