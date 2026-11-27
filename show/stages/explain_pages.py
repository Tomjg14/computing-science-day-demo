"""Stage 2 explanation (key E in free detection, the first two pages): how the old AI and the new
AI look at a face, live, with real numbers. The third page is the grid page (detect_explain.py).

Page "old": how the Haar cascade searches. Its window slides over the gray camera image
(slowed down, an illustration) next to the real windows where it finds a face. On the face, the
window becomes a gray 24 x 24 grid; the tests of stage 1 of the real cascade file run on it one by
one (white part counts plus, black part counts minus), then the page shows how many of the 25
stages the face passes. The simple version shows two real tests as yes/no questions on the face
(the strip over the eyes, the bridge of the nose), the 25 rounds and the search.

Page "new": a crop of the face goes into a copy of the YOLO face model. The page shows real
outputs of three layers and the face score of that copy. Simple: one filter per layer, and
the first one is an edge filter (measured, explain.FirstLayerFilters.edge_filters). In later
layers the maps get smaller (10 x 10 squares in the last one shown), so each square sees a
large part of the face; the page picks the filters whose output is highest on the face compared
with around it (explain.face_contrast), averaged over the first CHOOSE_TIME seconds, and the
simple page lays those maps over the face.

Simple or detailed: config.EXPLAIN_DETAIL (Shift+E). Both pages follow the explanation face
(facelock.py): it never switches by itself; when it is covered, the pages keep its last place,
so the effect of covering stays visible.
"""
import cv2
import numpy as np
import pygame

from .. import config
from ..draw import CYAN, GREEN, GREY, RED, WHITE, YELLOW, mix
from ..explain import (MAPS_PER_LAYER, FEATURE_LAYERS, FaceLayers, HaarCascadeFile, best_edge_filter,
                       cut_window, face_contrast, haar_window, padded_square, shared_first_layer)
from ..texts import t
from .filter_page import edge_strength_rgb
from .page_parts import STEPS, blit_image, draw_scan, draw_steps, score_bar

TEST_TIME = 1.6        # seconds per Haar test on screen
BLOCK_TIME = 2.0       # step 1: the zoomed block of numbers moves this often
HAAR_EVERY = 0.25      # seconds between two cascade evaluations (it runs in Python)
SCAN_EVERY = 0.5       # seconds between two searches of the whole image with the cascade
SCAN_WIDTH = 320       # the search runs on the camera image made this wide
QUESTION_TIME = 4.0    # simple old page: seconds per yes/no question
FACE_SIZE = 240        # simple old page: the gray face window on screen (pixels at 720 high)
CHOOSE_TIME = 1.5      # new page: seconds of model results that the filter choice averages over
CHOOSE_MIN = 3         # new page: at least this many results before the choice
SCALE_FOLLOW = 0.15    # new page: how fast the color scale follows each new result (0-1)


class ModelExplainer:
    def __init__(self):
        self.cascade = HaarCascadeFile()
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + config.HAAR_FILE)
        self.layers = FaceLayers()
        self.page = None           # None | "old" | "new"
        self.face_box, self.face_seen = None, 0.0
        self.window = self.result = None   # old page: (window box, found by Haar), cascade result
        self.grid = None                   # old page: the 24 x 24 gray window
        self.last_haar = 0.0
        self.channels = {}                 # new page: layer -> all filters, best first, fixed per opening
        self.edge = None                   # new page, simple: the edge filter shown for layer 1
        self.scale = {}                    # simple: layer -> (lo, hi) of its filter, follows the results
        self.dscale = {}                   # detailed: layer -> value that is full color, follows the results
        self.rank_sum, self.rank_n = {}, 0  # new page: summed face rankings before the choice
        self.rank_start, self.rank_res = None, None   # time of the first result, last result counted
        self.first = shared_first_layer()  # to find the edge filters of layer 1
        self.face_gray = None              # old page: the gray face window, large
        self.scan, self.last_scan = None, 0.0   # old page: (small gray image, found windows)

    def preload(self):
        self.layers.preload()
        self.first.preload()

    def open(self, page, now):
        """page: "old", "new" or None (closed)."""
        self.page = page
        self.channels, self.scale, self.dscale, self.edge = {}, {}, {}, None
        self.rank_sum, self.rank_n, self.rank_start, self.rank_res = {}, 0, None, None
        self.layers.result = None

    def close(self):
        self.page = None

    # --- data ---
    def update(self, app, det, now):
        if self.page is None or app.frame is None:
            return
        if self.page == "old" and now - self.last_scan >= SCAN_EVERY:
            self.last_scan = now
            fh, fw = app.frame.shape[:2]
            small = cv2.resize(app.frame, (SCAN_WIDTH, round(SCAN_WIDTH * fh / fw)), interpolation=cv2.INTER_AREA)
            small = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
            found = self.haar.detectMultiScale(small, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
            self.scan = (small, [tuple(int(v) for v in b) for b in found])
        self.face_box = app.face_lock.box
        if self.face_box is None:
            self.result = self.grid = self.face_gray = None
            return
        if self.page == "old" and now - self.last_haar >= HAAR_EVERY:
            self.last_haar = now
            gray = cv2.equalizeHist(cv2.cvtColor(app.frame, cv2.COLOR_BGR2GRAY))  # as the live cascade does
            self.window = haar_window(gray, self.face_box, self.haar)
            self.grid = cut_window(gray, self.window[0], self.cascade.size)
            self.face_gray = cut_window(gray, self.window[0], FACE_SIZE)
            self.result = self.cascade.evaluate(self.grid) if self.grid is not None else None
        elif self.page == "new":
            self.layers.submit(*padded_square(app.frame, self.face_box, margin=0.35))
            res = self.layers.result
            if res is not None and res is not self.rank_res:   # each model result counts once
                self.rank_res = res
                self._choose_filters(res, now)
                self._follow_scale(res)

    def _choose_filters(self, res, now):
        """Average the face ranking over the first CHOOSE_TIME seconds of results, then fix the
        order of the filters. One result is not enough: in one image a filter can win by chance
        and then react to the shoulders or clothes in the next images."""
        if self.channels:
            return
        if self.rank_start is None:
            self.rank_start = now
        for layer, maps in res["maps"].items():
            rank = self._ranking(maps, res)
            self.rank_sum[layer] = self.rank_sum.get(layer, 0) + rank
        self.rank_n += 1
        if now - self.rank_start < CHOOSE_TIME or self.rank_n < CHOOSE_MIN:
            return
        for layer, total in self.rank_sum.items():
            self.channels[layer] = [int(c) for c in np.argsort(-total)]
        first = FEATURE_LAYERS[0]
        if self.first.ready and first in res["maps"]:
            self.edge = best_edge_filter(self.first, res["maps"][first])

    def _follow_scale(self, res):
        """The color scale follows the results a little each time, so the face does not go dark
        when the values change after the choice."""
        if not self.channels:
            return
        for layer, maps in res["maps"].items():
            m = maps[self.channels[layer][0]]
            lo = float(m.min())
            target = (lo, max(lo + 1e-6, float(np.percentile(m, 99))))
            old = self.scale.get(layer, target)
            self.scale[layer] = tuple(o + SCALE_FOLLOW * (n - o) for o, n in zip(old, target))
            top = max(1e-6, float(maps[self.channels[layer][:MAPS_PER_LAYER]].max()))
            old = self.dscale.get(layer, top)
            self.dscale[layer] = old + SCALE_FOLLOW * (top - old)

    # --- drawing ---
    def draw(self, r, app, now):
        sw, sh = r.size
        r.shade(pygame.Rect(0, 0, sw, sh), 235)
        prefix = "x3o" if self.page == "old" else "x3n"
        detail = config.EXPLAIN_DETAIL
        x, y, w = r.u(40), r.top + r.u(24), r.u(500)
        n = self.layers.n_filters
        steps = [s.format(n=self.cascade.n_tests, f=f"{n:,}".replace(",", t("thousands")) if n else "")
                 for s in t(prefix + ("_steps" if detail else "s_steps"))]
        _, step = draw_steps(r, t(prefix + "_title"), steps, x, y, w, size=19 if detail else 22)
        r.text(t("x3_next" if self.page == "old" else "x3_next_grid"), 18, GREY, bottomleft=(x, sh - r.u(20)))

        area = pygame.Rect(sw - r.u(720), r.top + r.u(24), r.u(690), sh - r.top - r.u(48))
        if self.page == "old" and (step <= 0 if detail else step == 3) and self.scan is not None:
            self._draw_search(r, area, now)    # the search needs no face
            return
        if self.face_box is None:
            r.text(t("x3_no_face"), 28, YELLOW, center=area.center)
            return
        if not detail:
            self._draw_old_simple(r, area, step, now) if self.page == "old" else self._draw_new_simple(r, area, step)
        elif self.page == "old":
            self._draw_old(r, area, step, now)
        else:
            self._draw_new(r, area)

    def _draw_search(self, r, area, now):
        gray, found = self.scan
        w = area.w
        h = round(w * gray.shape[0] / gray.shape[1])
        draw_scan(r, pygame.Rect(area.x, area.y, w, min(h, area.h - r.u(60))), gray, found, now)

    # --- simple pages ---
    def _test_cells(self, test):
        """Per cell of the 24 x 24 window: +1 = must be light for a face, -1 = must be dark, 0 = not
        used. A test gives face points on one side of its limit; that side decides light or dark."""
        n = self.cascade.size
        w = np.zeros((n, n))
        for x, y, ww, hh, wt in test["rects"]:
            w[y:y + hh, x:x + ww] += wt
        face_high = test["points_right"] > test["points_left"]   # face when the value is above the limit
        return np.sign(w) * (1 if face_high else -1)

    def _draw_old_simple(self, r, area, step, now):
        """Two real yes/no questions on the face, the 25 rounds, and (step 4) the search."""
        size = r.u(FACE_SIZE)
        face = pygame.Rect(area.x, area.y, size, size)
        if self.face_gray is None or self.result is None:
            r.text(t("x3_no_face"), 28, YELLOW, center=area.center)
            return
        blit_image(r, self.face_gray, face)
        pygame.draw.rect(r.screen, WHITE, face.inflate(4, 4), 2)
        r.text(t("x3os_gray"), 15, GREY, bold=False, topleft=(face.x, face.bottom + r.u(6)))
        qx, qw = face.right + r.u(30), area.right - face.right - r.u(30)
        tests = self.result["stages"][0]["tests"]
        if step >= 1:
            q = int(STEPS.age(1, now) / QUESTION_TIME) % 2      # the first two tests of the real file
            test = tests[q]
            cells = self._test_cells(test)
            n, c = self.cascade.size, size / self.cascade.size
            tint = pygame.Surface((int(c) + 1, int(c) + 1), pygame.SRCALPHA)
            for row in range(n):
                for col in range(n):
                    if cells[row, col]:
                        tint.fill((255, 255, 255, 120) if cells[row, col] > 0 else (0, 0, 0, 150))
                        r.screen.blit(tint, (face.x + col * c, face.y + row * c))
            for x, y, ww, hh, _ in test["rects"]:   # the rectangles of the test, also on a light face
                pygame.draw.rect(r.screen, YELLOW, (face.x + x * c, face.y + y * c, ww * c, hh * c), r.u(2))
            y = area.y
            for line in r.wrap(t("x3os_q")[q], 22, qw):
                r.text(line, 22, WHITE, topleft=(qx, y))
                y += r.u(29)
            yes = test["points"] > 0
            r.text(t("x3os_yes") if yes else t("x3os_no"), 48, GREEN if yes else RED, topleft=(qx, y + r.u(6)))
            y += r.u(70)
            for swatch, key in (((235, 235, 235), "x3os_light"), ((20, 20, 20), "x3os_dark")):
                pygame.draw.rect(r.screen, swatch, (qx, y, r.u(22), r.u(22)))
                pygame.draw.rect(r.screen, GREY, (qx, y, r.u(22), r.u(22)), 1)
                r.text(t(key), 16, GREY, bold=False, midleft=(qx + r.u(32), y + r.u(11)))
                y += r.u(30)
        if step >= 2:
            y = face.bottom + r.u(44)
            passed, total = self.result["passed"], self.result["n_stages"]
            r.text(t("x3o_stages", k=passed, n=total), 22, WHITE, topleft=(area.x, y))
            box = r.u(24)
            for s in range(total):
                color = GREEN if s < passed else (RED if s == passed else (70, 70, 70))
                pygame.draw.rect(r.screen, color, (area.x + s * (box + r.u(3)), y + r.u(36), box, box),
                                 border_radius=r.u(4))
            ok = self.result["face"]
            r.text(t("x3o_face") if ok else t("x3o_noface"), 34, GREEN if ok else RED,
                   topleft=(area.x, y + r.u(74)))

    def _draw_new_simple(self, r, area, step):
        """The face, one real output per layer (layer 1: an edge filter), and the face score."""
        res = self.layers.result
        if res is None or not self.channels:
            r.text(t("x3n_wait"), 26, YELLOW, center=area.center)
            return
        size = r.u(170)
        face = pygame.Rect(area.x, area.y, size, size)
        blit_image(r, res["crop"], face)
        r.text(t("x3n_input"), 15, GREY, bold=False, topleft=(face.x, face.bottom + r.u(6)))
        tile, gap = r.u(190), r.u(40)
        y = face.bottom + r.u(50)
        layers = list(res["maps"].items())
        bottom = y
        for j, (layer, maps) in enumerate(layers):
            if j > 0 and step < 1:
                break
            # layer 1: the edge filter; later layers: the filter that points at the face most clearly
            ch = self.edge if j == 0 and self.edge is not None else self.channels[layer][0]
            m = maps[ch]
            rect = pygame.Rect(area.x + j * (tile + gap), y, tile, tile)
            if j == 0:   # the edge filter: edges bright, flat areas dark, like the edge image of step 1
                blit_image(r, edge_strength_rgb(m), rect)
            else:        # a see-through color layer on the face: where this filter reacts strongly
                lo, hi = self.scale[layer]
                heat = cv2.applyColorMap((np.clip((m - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8),
                                         cv2.COLORMAP_INFERNO)
                heat = cv2.cvtColor(cv2.resize(heat, (tile, tile), interpolation=cv2.INTER_NEAREST),
                                    cv2.COLOR_BGR2RGB)        # the squares stay visible as blocks
                photo = cv2.resize(res["crop"], (tile, tile), interpolation=cv2.INTER_AREA)
                blit_image(r, cv2.addWeighted(photo, 0.45, heat, 0.55, 0), rect)
            if j > 0:
                r.text("→", 34, WHITE, center=(rect.x - gap // 2, rect.centery))
            ly = rect.bottom + r.u(6)
            for line in r.wrap(t("x3ns_layer")[j].format(n=FEATURE_LAYERS[j] + 1, s=m.shape[0]), 16,
                               tile + gap - r.u(10), bold=False):
                r.text(line, 16, GREY, bold=False, topleft=(rect.x, ly))
                ly += r.u(21)
            bottom = ly
        if step >= 1:
            ny = bottom + r.u(10)
            for line in r.wrap(t("x3ns_note"), 17, area.w, bold=False):
                r.text(line, 17, WHITE, bold=False, topleft=(area.x, ny))
                ny += r.u(23)
        if step >= 2:
            sx = face.right + r.u(30)
            r.text(t("x3n_score"), 22, WHITE, topleft=(sx, area.y + r.u(10)))
            score_bar(r, pygame.Rect(sx, area.y + r.u(50), area.right - sx, r.u(40)), res["conf"],
                      config.FACE_VISIBLE_CONF)
            yy = area.y + r.u(104)
            for line in r.wrap(t("x3n_real"), 16, area.right - sx, bold=False):
                r.text(line, 16, CYAN, bold=False, topleft=(sx, yy))
                yy += r.u(21)

    @staticmethod
    def _ranking(maps, res):
        """Per filter: how clearly it points at the face (higher on the face than around it)."""
        return face_contrast(maps, res["face"]) if res.get("face") else maps.mean(axis=(1, 2))

    def _draw_old(self, r, area, newest, now):
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

        step = newest - 1   # which text step is on screen, after the search step (0 = numbers)
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
            i = int(STEPS.age(2, now) / TEST_TIME) % len(stage1["tests"])
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
        for line in r.wrap(t("x3o_stage1", p=f"{stage1['total']:.2f}", lim=f"{stage1['limit']:.2f}"), 22,
                           area.right - info_x):
            r.text(line, 22, GREEN if stage1["ok"] else RED, topleft=(info_x, y))
            y += r.u(30)
        y += r.u(14)
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
        if res is None or not self.channels:
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
            r.text(t("x3n_layer", n=layer + 1, c=maps.shape[0], k=MAPS_PER_LAYER, s=maps.shape[1]), 18, WHITE,
                   topleft=(area.x, y))
            y += r.u(26)
            for j, ch in enumerate(self.channels[layer][:MAPS_PER_LAYER]):
                m = np.clip(maps[ch] / self.dscale[layer], 0, 1)
                heat = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
                heat = cv2.cvtColor(cv2.resize(heat, (tile, tile), interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB)
                pos = (area.x + j * (tile + r.u(12)), y)
                r.screen.blit(pygame.image.frombuffer(np.ascontiguousarray(heat).tobytes(), (tile, tile), "RGB"), pos)
            y += tile + r.u(14)
        y = area.y + r.u(108)
        for line in r.wrap(t("x3n_real"), 18, area.right - score_x, bold=False):
            r.text(line, 18, CYAN, bold=False, topleft=(score_x, y))
            y += r.u(24)
