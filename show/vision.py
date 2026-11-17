"""Camera input and the two detection models (faces, and objects including persons)."""
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import requests
from ultralytics import YOLO

from . import config


@dataclass
class Face:
    id: object               # track id, or None
    box: tuple               # x1, y1, x2, y2 in frame pixels
    conf: float
    kpts: object = None      # (5, 3) array: eyes, nose, mouth corners with confidence

    @property
    def width(self):
        return self.box[2] - self.box[0]


@dataclass
class Obj:
    id: object
    name: str                # English class name
    box: tuple
    conf: float


@dataclass
class Detections:
    frame_w: int
    frame_h: int
    faces: list = field(default_factory=list)
    objects: list = field(default_factory=list)   # everything except persons, above OBJECT_CONF
    all_objects: list = field(default_factory=list)  # also weak guesses (from OBJECT_MIN_CONF), for meters
    persons: list = field(default_factory=list)
    face_ms: float = 0.0
    object_ms: float = 0.0


def _model_path(name, url=None):
    path = config.ROOT / name
    if path.exists() and path.stat().st_size > 1_000_000:
        return str(path)
    if url is None:
        return name  # let ultralytics download it
    print(f"[SHOW] Downloading {name} ...")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)
    return str(path)


class VisionEngine:
    def __init__(self):
        self.face_model = YOLO(_model_path(config.FACE_MODEL, config.FACE_MODEL_URL))
        self.object_model = YOLO(_model_path(config.OBJECT_MODEL))
        self.names = self.object_model.names

    def process(self, frame):
        h, w = frame.shape[:2]
        det = Detections(frame_w=w, frame_h=h)

        t0 = time.time()
        r = self.face_model.track(frame, persist=True, verbose=False, conf=config.FACE_DETECT_CONF)[0]
        kpts = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
        for i, (tid, box, conf) in enumerate(_boxes(r)):
            det.faces.append(Face(tid, box, conf, kpts[i] if kpts is not None and i < len(kpts) else None))

        t1 = time.time()
        r = self.object_model.track(frame, persist=True, verbose=False, conf=config.OBJECT_MIN_CONF)[0]
        classes = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
        for (tid, box, conf), cls in zip(_boxes(r), classes):
            o = Obj(tid, self.names[int(cls)], box, conf)
            if o.name != "person":
                det.all_objects.append(o)
            if conf >= config.OBJECT_CONF:
                (det.persons if o.name == "person" else det.objects).append(o)
        t2 = time.time()

        det.face_ms, det.object_ms = (t1 - t0) * 1000, (t2 - t1) * 1000
        return det


def _boxes(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else [None] * len(xyxy)
    return [(tid, tuple(float(v) for v in b), float(c)) for tid, b, c in zip(ids, xyxy, conf)]


class Camera:
    """Webcam with its own reader thread, so the models never wait for the camera:
    read() returns the newest image, mirrored."""

    def __init__(self, index):
        self.index = index
        self._cap_lock = threading.Lock()
        self._lock = threading.Lock()
        self._frame, self._count, self._taken = None, 0, 0
        self.fps = 0.0
        self.cap = self._open(index)
        if self.cap is None:
            self.next()
        if self.cap is None:
            raise SystemExit("No working camera found.")
        self._running = True
        threading.Thread(target=self._reader, daemon=True).start()

    @staticmethod
    def _open(index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # 30 fps on most USB webcams
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
        if cap.isOpened() and cap.read()[0]:
            return cap
        cap.release()
        return None

    def _reader(self):
        last = time.time()
        while self._running:
            with self._cap_lock:
                ok, frame = self.cap.read() if self.cap is not None else (False, None)
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._count += 1
            now = time.time()
            self.fps = 0.8 * self.fps + 0.2 / max(1e-3, now - last)
            last = now

    def next(self):
        """Switch to the next camera that works; keep the old one if none does."""
        with self._cap_lock:
            if self.cap is not None:
                self.cap.release()
            for step in range(1, config.MAX_CAMERAS + 1):
                idx = (self.index + step) % config.MAX_CAMERAS
                cap = self._open(idx)
                if cap is not None:
                    self.index, self.cap = idx, cap
                    return
            self.cap = self._open(self.index)

    def peek(self):
        """(image number, newest raw image) for the screen; does not take it from the AI thread."""
        with self._lock:
            return self._count, self._frame

    def read(self, timeout=0.5):
        """The newest image that was not returned before (waits briefly), or None."""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self._count != self._taken and self._frame is not None:
                    self._taken = self._count
                    return cv2.flip(self._frame, 1)  # mirror view feels natural
            time.sleep(0.002)
        return None

    def release(self):
        self._running = False
        time.sleep(0.05)
        with self._cap_lock:
            if self.cap is not None:
                self.cap.release()


class Segmenter:
    """Segmentation model for the silhouette game. It loads in the background and runs once
    per freeze, so it never slows down the live detection."""

    def __init__(self):
        self.model = None
        self._lock = threading.Lock()

    def preload(self):
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        with self._lock:
            if self.model is None:
                self.model = YOLO(_model_path(config.SEG_MODEL))

    def run_async(self, frame, done):
        """Find objects with their exact shape; calls done(list of dicts) from a thread."""
        def work():
            self._load()
            r = self.model.predict(frame, conf=config.SEG_CONF, retina_masks=True, verbose=False)[0]
            found = []
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy() > 0.5
                for (_, box, conf), cls, mask in zip(_boxes(r), r.boxes.cls.cpu().numpy().astype(int), masks):
                    name = self.model.names[int(cls)]
                    if name != "person":
                        found.append({"name": name, "conf": conf, "box": box, "mask": mask})
            done(found)
        threading.Thread(target=work, daemon=True).start()


class VisionThread(threading.Thread):
    """Reads the camera and runs the models in the background, so the screen can draw at 60 fps.

    Each result pairs a frame with its detections, so boxes always match the image.
    """

    def __init__(self, engine, camera):
        super().__init__(daemon=True)
        self.engine, self.camera = engine, camera
        self._lock = threading.Lock()
        self._latest = None
        self._seq = 0
        self._running = True
        self._switch = False
        self.fps = 0.0

    def run(self):
        last = time.time()
        while self._running:
            if self._switch:
                self.camera.next()
                self._switch = False
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            det = self.engine.process(frame)
            with self._lock:
                self._latest = (frame, det)
                self._seq += 1
            now = time.time()
            self.fps = 0.8 * self.fps + 0.2 / max(1e-3, now - last)
            last = now

    def latest(self):
        """(sequence number, (frame, detections) or None)."""
        with self._lock:
            return self._seq, self._latest

    def switch_camera(self):
        self._switch = True

    def stop(self):
        self._running = False
        self.join(timeout=2)
        self.camera.release()


def largest_face(faces, min_conf):
    faces = [f for f in faces if f.conf >= min_conf]
    return max(faces, key=lambda f: f.width, default=None)


def crop_square(frame, box, margin=0.25):
    """Square crop around a box, with some margin, clipped to the frame."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (0.5 + margin)
    h, w = frame.shape[:2]
    a, b = int(max(0, cx - half)), int(max(0, cy - half))
    c, d = int(min(w, cx + half)), int(min(h, cy + half))
    return np.ascontiguousarray(frame[b:d, a:c])
