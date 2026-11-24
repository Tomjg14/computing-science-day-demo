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
                                "points": points, "low": value < tlimit})
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


class FaceLayers:
    """A second copy of the face model, for the explanation only. It runs in its own thread,
    one face crop at a time, so the show never waits for it."""

    def __init__(self):
        self.model = None
        self.result = None          # {"crop": rgb, "maps": {layer: array (channels, h, w)}, "conf": float}
        self._busy = False
        self._lock = threading.Lock()
        self._outs = {}

    def _load(self):
        with self._lock:
            if self.model is None:
                from ultralytics import YOLO
                from .vision import _model_path
                self.model = YOLO(_model_path(config.FACE_MODEL, config.FACE_MODEL_URL))
                net = self.model.model.model
                for i in FEATURE_LAYERS:
                    net[i].register_forward_hook(lambda mod, inp, out, i=i: self._outs.__setitem__(i, out))

    def preload(self):
        threading.Thread(target=self._load, daemon=True).start()

    def submit(self, crop_bgr):
        """Run the model on this crop, unless it is still busy with the previous one."""
        if self._busy or crop_bgr is None:
            return
        self._busy = True

        def work():
            try:
                self._load()
                r = self.model.predict(crop_bgr, imgsz=320, verbose=False, conf=0.05)[0]
                conf = float(r.boxes.conf.max()) if r.boxes is not None and len(r.boxes) else 0.0
                maps = {i: self._outs[i][0].detach().float().cpu().numpy() for i in FEATURE_LAYERS}
                self.result = {"crop": cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), "maps": maps, "conf": conf}
            finally:
                self._busy = False
        threading.Thread(target=work, daemon=True).start()


class ObjectGrid:
    """A second copy of the object model, for the stage 2 explanation. It looks at a smaller copy
    of the camera image (sides a multiple of 32, so there is no padding) and keeps, for every
    square of the grid, the scores of all 80 object types (sigmoid of the model's raw output).
    YOLO uses three grid sizes (squares of 8, 16 and 32 pixels); the coarse grid of 32 pixels
    gets, per square, the highest score of all squares inside it."""

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
            centers, start = [], 0
            for (rh, rw) in sizes:
                n = rh * rw
                best = prob[:, start:start + n].max(axis=0).reshape(rh, rw)
                f = rh // gh
                heat = np.maximum(heat, best.reshape(gh, f, gw, f).max(axis=(1, 3)))
                s = h / rh
                ys, xs = np.mgrid[0:rh, 0:rw]
                centers.append(np.stack([(xs.ravel() + 0.5) * s, (ys.ravel() + 0.5) * s], axis=1))
                start += n
            centers = np.concatenate(centers)
            sx, sy = fw / w, fh / h
            dets = [(self.model.names[int(c)], float(p), (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy))
                    for c, p, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
            focus = None
            i = pick(dets) if dets else None
            if i is not None:
                name, _, (x1, y1, x2, y2) = dets[i]
                cls = next(k for k, v in self.model.names.items() if v == name)
                cx, cy = centers[:, 0] * sx, centers[:, 1] * sy
                inside = np.where((cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2))[0]
                if len(inside):
                    a = inside[np.argmax(prob[cls, inside])]           # the square that is surest of this type
                    top = np.argsort(-prob[:, a])[:5]
                    focus = {"det": i, "center": (cx[a], cy[a]),
                             "top": [(self.model.names[int(k)], float(prob[k, a])) for k in top]}
            self.result = {"image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "heat": heat,
                           "frame_size": (fw, fh), "dets": dets, "focus": focus}
        finally:
            self._busy = False
