"""Stage 3 explanation (key E): how the old AI and the new AI look at a face, live, with real numbers.

Page "old": the face becomes a gray 24 x 24 grid, as the Haar cascade sees it. The tests of
stage 1 of the real cascade file run on it one by one (white part counts plus, black part
counts minus), then the page shows how many of the 25 stages the face passes.

Page "new": a crop of the face goes into a copy of the YOLO face model. The page shows real
outputs of three layers (4 filters each) and the face score of that copy.

Both pages follow the largest face; when it disappears (for example covered), they keep its
last place for a few seconds, so the effect of covering stays visible.
"""
import cv2
import numpy as np
import pygame

from .. import config
from ..draw import BLACK, CYAN, GREEN, GREY, RED, WHITE, YELLOW, mix
from ..explain import MAPS_PER_LAYER, FaceLayers, HaarCascadeFile, cut_window, haar_window
from ..texts import t
from ..vision import crop_square

STEP_TIME = 4.0        # seconds per text step
TEST_TIME = 1.6        # seconds per Haar test on screen
BLOCK_TIME = 2.0       # step 1: the zoomed block of numbers moves this often
KEEP_FACE = 3.0        # seconds the last face place is kept when the face is gone
HAAR_EVERY = 0.25      # seconds between two cascade evaluations (it runs in Python)


class ModelExplainer:
    def __init__(self):
        self.cascade = HaarCascadeFile()
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + config.HAAR_FILE)
        self.layers = FaceLayers()
        self.page = None           # None | "old" | "new"
        self.start = 0.0
        self.face_box, self.face_seen = None, 0.0
        self.window = self.result = None   # old page: (window box, found by Haar), cascade result
        self.grid = None                   # old page: the 24 x 24 gray window
        self.last_haar = 0.0
        self.channels = {}                 # new page: layer -> chosen channels, fixed per opening
        self.scale = {}                    # layer -> value that is full color

    def preload(self):
        self.layers.preload()

    def next_page(self, now):
        """E: off -> old -> new -> off."""
        self.page = {None: "old", "old": "new", "new": None}[self.page]
        self.start = now
        self.channels, self.scale = {}, {}
        self.layers.result = None

    def close(self):
        self.page = None

    # --- data ---
    def update(self, app, det, now):
        if self.page is None or app.frame is None:
            return
        faces = [f for f in det.faces if f.conf >= config.FACE_VISIBLE_CONF]
        if faces:
            self.face_box, self.face_seen = max(faces, key=lambda f: f.width).box, now
        elif now - self.face_seen > KEEP_FACE:
            self.face_box = None
        if self.face_box is None:
            self.result = self.grid = None
            return
        if self.page == "old" and now - self.last_haar >= HAAR_EVERY:
            self.last_haar = now
            gray = cv2.equalizeHist(cv2.cvtColor(app.frame, cv2.COLOR_BGR2GRAY))  # as the live cascade does
            self.window = haar_window(gray, self.face_box, self.haar)
            self.grid = cut_window(gray, self.window[0], self.cascade.size)
            self.result = self.cascade.evaluate(self.grid) if self.grid is not None else None
        elif self.page == "new":
            self.layers.submit(crop_square(app.frame, self.face_box, margin=0.35))

    # --- drawing ---
    def draw(self, r, app, now):
        sw, sh = r.size
        r.shade(pygame.Rect(0, 0, sw, sh), 235)
        el = now - self.start
        prefix = "x3o" if self.page == "old" else "x3n"
        x, y, w = r.u(40), r.top + r.u(24), r.u(500)
        r.text(t(prefix + "_title"), 32, CYAN, topleft=(x, y))
        y += r.u(54)
        steps = t(prefix + "_steps")
        shown = min(len(steps), 1 + int(el / STEP_TIME))
        for i in range(shown):
            circle = (x + r.u(15), y + r.u(15))
            pygame.draw.circle(r.screen, YELLOW, circle, r.u(15))
            r.text(str(i + 1), 20, BLACK, shadow=False, center=circle)
            for line in r.wrap(steps[i], 22, w - r.u(44)):
                r.text(line, 22, WHITE, topleft=(x + r.u(44), y))
                y += r.u(29)
            y += r.u(12)
        if self.page == "new" and shown == len(steps):
            from .detect_explain import draw_yolo_note
            draw_yolo_note(r, x, y + r.u(8), w)
        r.text(t("x3_next" if self.page == "old" else "x3_close"), 18, GREY, bottomleft=(x, sh - r.u(20)))

        area = pygame.Rect(sw - r.u(720), r.top + r.u(24), r.u(690), sh - r.top - r.u(48))
        if self.face_box is None:
            r.text(t("x3_no_face"), 28, YELLOW, center=area.center)
            return
        if self.page == "old":
            self._draw_old(r, area, el, now)
        else:
            self._draw_new(r, area)

    def _draw_old(self, r, area, el, now):
        if self.grid is None or self.result is None:
            r.text(t("x3_no_face"), 28, YELLOW, center=area.center)
            return
        n = self.cascade.size
        cell = r.u(16)
        gx, gy = area.x, area.y
        for row in range(n):
            for col in range(n):
                v = int(self.grid[row, col])
                pygame.draw.rect(r.screen, (v, v, v), (gx + col * cell, gy + row * cell, cell, cell))
        pygame.draw.rect(r.screen, WHITE, (gx - 3, gy - 3, n * cell + 6, n * cell + 6), 2)
        info_x, info_y = gx + n * cell + r.u(24), gy
        r.text(t("x3o_grid"), 18, GREY, bold=False, topleft=(gx, gy + n * cell + r.u(10)))

        step = int(el / STEP_TIME)   # which text step is on screen (0 = the first)
        if step < 1:
            # step 1: a block of the grid as numbers, like stage 1
            k = 6
            bi = int(now / BLOCK_TIME) % 3
            br, bc = [(6, 3), (6, 14), (15, 9)][bi]
            pygame.draw.rect(r.screen, YELLOW, (gx + bc * cell, gy + br * cell, k * cell, k * cell), r.u(3))
            for i in range(k):
                for j in range(k):
                    r.text(str(int(self.grid[br + i, bc + j])), 18, WHITE,
                           topleft=(info_x + j * r.u(44), info_y + r.u(10) + i * r.u(30)))
            return

        stage1 = self.result["stages"][0]
        if step < 3:
            # steps 2 and 3: the tests of stage 1, one at a time
            i = int((el - STEP_TIME) / TEST_TIME) % len(stage1["tests"])
            test = stage1["tests"][i]
            weights = np.zeros((n, n))
            for x, y, w, h, wt in test["rects"]:
                weights[y:y + h, x:x + w] += wt
            tint = pygame.Surface((cell, cell), pygame.SRCALPHA)
            for row in range(n):
                for col in range(n):
                    if weights[row, col] != 0:
                        tint.fill((255, 255, 255, 150) if weights[row, col] > 0 else (0, 0, 0, 170))
                        r.screen.blit(tint, (gx + col * cell, gy + row * cell))
            plus = float(self.grid[weights > 0].astype(int).sum())
            minus = float(self.grid[weights < 0].astype(int).sum())
            lines = [(t("x3o_test", i=i + 1, n=len(stage1["tests"])), YELLOW, 24),
                     (t("x3o_plus", s=f"{plus:.0f}"), WHITE, 20),
                     (t("x3o_minus", s=f"{minus:.0f}"), WHITE, 20),
                     (t("x3o_value", v=f"{test['value']:+.3f}", lim=f"{test['limit']:+.3f}"), CYAN, 20),
                     (t("x3o_points", p=f"{test['points']:+.2f}"), GREEN if test["points"] > 0 else RED, 26)]
            y = info_y
            for s, color, size in lines:
                for line in r.wrap(s, size, area.right - info_x):
                    r.text(line, size, color, topleft=(info_x, y))
                    y += r.u(size + 10)
                y += r.u(6)
            return

        # steps 4 and 5: stage 1 total, then the 25 stages
        y = info_y
        r.text(t("x3o_stage1", p=f"{stage1['total']:.2f}", lim=f"{stage1['limit']:.2f}"), 22,
               GREEN if stage1["ok"] else RED, topleft=(info_x, y))
        y += r.u(44)
        r.text(t("x3o_stages", k=self.result["passed"], n=self.result["n_stages"]), 22, WHITE, topleft=(info_x, y))
        y += r.u(36)
        box = r.u(22)
        for s in range(self.result["n_stages"]):
            color = GREEN if s < self.result["passed"] else (RED if s == self.result["passed"] else (70, 70, 70))
            pygame.draw.rect(r.screen, color, (info_x + (s % 5) * (box + r.u(6)), y + (s // 5) * (box + r.u(6)),
                                               box, box), border_radius=r.u(4))
        y += 5 * (box + r.u(6)) + r.u(16)
        face = self.result["face"]
        r.text(t("x3o_face") if face else t("x3o_noface"), 30, GREEN if face else RED, topleft=(info_x, y))

    def _draw_new(self, r, area):
        res = self.layers.result
        if res is None:
            r.text(t("x3n_wait"), 26, YELLOW, center=area.center)
            return
        crop = res["crop"]
        size = r.u(150)
        img = cv2.resize(crop, (size, size))
        r.screen.blit(pygame.image.frombuffer(np.ascontiguousarray(img).tobytes(), (size, size), "RGB"), area.topleft)
        r.text(t("x3n_input"), 18, GREY, bold=False, topleft=(area.x, area.y + size + r.u(6)))
        score_x = area.x + size + r.u(30)
        r.text(t("x3n_score"), 22, WHITE, topleft=(score_x, area.y + r.u(20)))
        bar = pygame.Rect(score_x, area.y + r.u(60), area.right - score_x - r.u(10), r.u(30))
        r.bar(bar, res["conf"], mix(RED, GREEN, res["conf"]))
        r.text(f"{res['conf']:.0%}", 20, WHITE, shadow=False, midleft=(bar.x + r.u(8), bar.centery))

        y = area.y + size + r.u(40)
        tile = r.u(100)
        for layer, maps in res["maps"].items():
            if layer not in self.channels:   # choose the strongest filters once per opening, then keep them
                order = np.argsort(-maps.mean(axis=(1, 2)))[:MAPS_PER_LAYER]
                self.channels[layer] = order
                self.scale[layer] = max(1e-6, float(maps[order].max()))
            r.text(t("x3n_layer", n=layer + 1, c=maps.shape[0], k=MAPS_PER_LAYER, s=maps.shape[1]), 18, WHITE,
                   topleft=(area.x, y))
            y += r.u(26)
            for j, ch in enumerate(self.channels[layer]):
                m = np.clip(maps[ch] / self.scale[layer], 0, 1)
                heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                heat = cv2.cvtColor(cv2.resize(heat, (tile, tile), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
                pos = (area.x + j * (tile + r.u(12)), y)
                r.screen.blit(pygame.image.frombuffer(np.ascontiguousarray(heat).tobytes(), (tile, tile), "RGB"), pos)
            y += tile + r.u(14)
        y = area.y + r.u(108)
        for line in r.wrap(t("x3n_real"), 18, area.right - score_x, bold=False):
            r.text(line, 18, CYAN, bold=False, topleft=(score_x, y))
            y += r.u(24)
