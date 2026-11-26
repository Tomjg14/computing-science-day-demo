"""Stage 1 explanation (key E after the photo): what the first layer of the AI does with the pixel
numbers, with a real filter of the face model and the real calculation (explain.first_layer).

The page uses the same still and the same big pixels (config.PIXEL_GRID) as stage 1, so the
numbers on screen are the ones the audience just saw. A 3 x 3 window jumps 2 pixels at a time,
as in the model; each place gives one pixel of a new image. The table shows the window: pixel
numbers (0-1, as the model gets them), weights, the sum per color, the fixed number and the last
step (SiLU). Then the same filter on all pixels of the photo, and 4 other filters of the 32.

The filter on screen is one that reacts to edges and hardly to light (measured, see
FirstLayerFilters.edge_filters); of those, the one with the most contrast on this photo.
"""
import cv2
import numpy as np
import pygame

from ..draw import CYAN, GREY, WHITE, YELLOW
from ..explain import best_edge_filter, first_layer, shared_first_layer
from ..texts import t
from .page_parts import STEPS, blit_image, draw_steps

SLIDE_TIME = 0.05      # seconds per window place while it slides over the grid
FULL_SIZE = 240        # the photo goes into the filter at this size (the output is half)
N_OTHERS = 4
CHANNEL_COLORS = [(255, 70, 70), (70, 230, 90), (80, 140, 255)]
PLUS, MINUS = (60, 200, 110), (230, 70, 70)


def scale(values):
    """The color scale of an output image: from the lowest value to the 99th percentile, inside the
    image (without the row and column at the border). The padding of the model makes the border
    strong, and a few very strong pixels would make the rest look dark; values above the scale
    show at full color. Only the colors change, not the numbers."""
    inner = values[1:-1, 1:-1] if min(values.shape) > 2 else values
    return float(inner.min()), float(np.percentile(inner, 99))


def heat_rgb(values, lo=None, hi=None):
    """Values to colors: dark = low, bright yellow = high (one scale per image)."""
    if lo is None:
        lo, hi = scale(values)
    m = np.clip((values - lo) / max(1e-9, hi - lo), 0, 1)
    img = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def surface(rgb, size):
    img = cv2.resize(np.ascontiguousarray(rgb), (size, size), interpolation=cv2.INTER_NEAREST)
    return pygame.image.frombuffer(np.ascontiguousarray(img).tobytes(), (size, size), "RGB").copy()


def edge_strength_rgb(values):
    """An edge filter output for the simple pages: how much each spot differs from a flat area
    (the most common value), so edges are bright and flat areas dark, like the edge image.
    Only the colors change, not the numbers of the model."""
    dev = np.abs(values - np.median(values))
    img = np.clip(dev / max(1e-9, float(np.percentile(dev, 99))) * 255, 0, 255).astype(np.uint8)
    return np.repeat(img[:, :, None], 3, axis=2)


def fmt_weight(w):
    return f"{w:+.0f}" if abs(w) >= 9.5 else f"{w:+.1f}"


class FilterExplainer:
    filters = None         # FirstLayerFilters, shared: it loads the model once

    def __init__(self):
        if FilterExplainer.filters is None:
            FilterExplainer.filters = shared_first_layer()
        self.on = False
        self.data = None

    def preload(self):
        self.filters.preload()

    def open(self, now, photo, cells):
        self.on = True
        self.data = None
        self._photo, self._cells = photo, cells

    def close(self):
        self.on = False

    def _compute(self):
        f = self.filters
        full_in = cv2.resize(self._photo, (FULL_SIZE, FULL_SIZE), interpolation=cv2.INTER_AREA) / 255.0
        full = first_layer(full_in, f.weights, f.bias)                    # (32, 120, 120)
        contrast = full.std(axis=(1, 2))
        k = best_edge_filter(f, full)
        others = [i for i in np.argsort(-contrast) if i != k][:N_OTHERS]
        x = self._cells / 255.0
        out = first_layer(x, f.weights[k:k + 1], f.bias[k:k + 1])[0]     # (g/2, g/2)
        inner = out[1:-1, 1:-1]                                             # no padding in the window
        fi, fj = np.unravel_index(np.argmax(np.abs(inner - np.median(out))), inner.shape)
        self.data = {"k": k, "x": x, "pad": np.pad(x, ((1, 1), (1, 1), (0, 0))), "out": out,
                     "focus": (fi + 1, fj + 1),
                     "full": full[k], "others": [full[i] for i in others], "n": len(f.weights)}

    # --- drawing ---
    def draw(self, r, now):
        sw, sh = r.size
        r.shade(pygame.Rect(0, 0, sw, sh), 235)
        x, y, w = r.u(40), r.top + r.u(24), r.u(470)
        n = len(self.filters.weights) if self.filters.ready else 32
        steps = [s.format(n=n) for s in t("x1_steps")]
        _, step = draw_steps(r, t("x1_title"), steps, x, y, w, size=21)
        r.text(t("x3_close"), 18, GREY, bottomleft=(x, sh - r.u(20)))

        area = pygame.Rect(sw - r.u(720), r.top + r.u(20), r.u(690), sh - r.top - r.u(40))
        if not self.filters.ready:
            r.text(t("x1_wait"), 26, YELLOW, center=area.center)
            return
        if self.data is None:
            self._compute()
        d = self.data
        n_out = d["out"].shape[0]
        places = n_out * n_out
        slide = int(STEPS.age(2, now) / SLIDE_TIME) if step >= 2 else -1   # from step 3 on
        if 0 <= slide < places:
            pos, shown = divmod(slide, n_out), slide + 1
        else:
            pos, shown = d["focus"], (places if slide >= places else 0)
        self._draw_grids(r, area, d, pos, shown, step)
        low = area.y + r.u(262)
        if step < 3:
            self._draw_table(r, pygame.Rect(area.x, low, area.w, area.bottom - low), d, pos, step)
        else:
            self._draw_full(r, pygame.Rect(area.x, low, area.w, area.bottom - low), d, step)
        r.text(t("x1_real"), 16, CYAN, bold=False, bottomright=(area.right, sh - r.u(14)))

    def _draw_grids(self, r, area, d, pos, shown, step):
        g = d["x"].shape[0]
        c = r.u(11)
        x0, y0 = area.x + c, area.y + c                      # one cell of room for the padding
        cells = (d["x"] * 255).astype(np.uint8)
        r.screen.blit(surface(cells, g * c), (x0, y0))
        pygame.draw.rect(r.screen, WHITE, (x0 - 2, y0 - 2, g * c + 4, g * c + 4), 1)
        oi, oj = pos
        win = pygame.Rect(x0 + (2 * oj - 1) * c, y0 + (2 * oi - 1) * c, 3 * c, 3 * c)
        pygame.draw.rect(r.screen, YELLOW, win, r.u(3))
        r.text(t("x1_in"), 16, GREY, bold=False, topleft=(area.x, y0 + g * c + r.u(8)))

        n = d["out"].shape[0]
        oc = r.u(22)
        ox = area.x + r.u(330)
        r.text("→", 40, WHITE, center=((x0 + g * c + ox) // 2, y0 + g * c // 2))
        pygame.draw.rect(r.screen, (40, 40, 40), (ox, y0, n * oc, n * oc))
        heat = heat_rgb(d["out"])
        show_focus = step >= 1 and shown == 0
        for i in range(n):
            for j in range(n):
                if i * n + j < shown or (show_focus and (i, j) == pos):
                    pygame.draw.rect(r.screen, tuple(int(v) for v in heat[i, j]), (ox + j * oc, y0 + i * oc, oc, oc))
        pygame.draw.rect(r.screen, WHITE, (ox - 2, y0 - 2, n * oc + 4, n * oc + 4), 1)
        cell = pygame.Rect(ox + oj * oc, y0 + oi * oc, oc, oc)
        if step >= 1 or shown:
            pygame.draw.rect(r.screen, YELLOW, cell, r.u(3))
            pygame.draw.line(r.screen, YELLOW, win.midright, cell.midleft, 1)
        r.text(t("x1_out"), 16, GREY, bold=False, topleft=(ox, y0 + g * c + r.u(8)))

    def _draw_table(self, r, area, d, pos, step):
        oi, oj = pos
        window = d["pad"][2 * oi:2 * oi + 3, 2 * oj:2 * oj + 3, :]          # (3, 3, colors), padded grid
        wts = self.filters.weights[d["k"]]                                   # (colors, 3, 3)
        wmax = np.abs(wts).max()
        c = r.u(33)
        row_h = r.u(108)
        sums = []
        for ch in range(3):
            y = area.y + ch * row_h
            r.text(t("x1_colors")[ch], 18, CHANNEL_COLORS[ch], midleft=(area.x, y + 3 * c // 2))
            px = area.x + r.u(62)
            for i in range(3):
                for j in range(3):
                    v = window[i, j, ch]
                    col = [0, 0, 0]
                    col[ch] = int(v * 255)
                    rect = pygame.Rect(px + j * c, y + i * c, c, c)
                    pygame.draw.rect(r.screen, col, rect)
                    pygame.draw.rect(r.screen, (90, 90, 90), rect, 1)
                    r.text(f"{v:.2f}", 12, WHITE, center=rect.center)
            s = float((window[:, :, ch] * wts[ch]).sum())
            sums.append(s)
            if step < 1:
                continue
            r.text("×", 26, WHITE, center=(px + 3 * c + r.u(16), y + 3 * c // 2))
            wx = px + 3 * c + r.u(32)
            for i in range(3):
                for j in range(3):
                    wv = wts[ch, i, j]
                    rect = pygame.Rect(wx + j * c, y + i * c, c, c)
                    tint = pygame.Surface((c, c), pygame.SRCALPHA)
                    tint.fill((*(PLUS if wv > 0 else MINUS), int(40 + 200 * abs(wv) / wmax)))
                    r.screen.blit(tint, rect.topleft)
                    pygame.draw.rect(r.screen, (90, 90, 90), rect, 1)
                    r.text(fmt_weight(wv), 12, WHITE, center=rect.center)
            r.text(t("x1_sum", v=f"{s:+.1f}"), 18, WHITE, midleft=(wx + 3 * c + r.u(12), y + 3 * c // 2))
        if step < 1:
            r.text(t("x1_scale"), 16, GREY, bold=False, topleft=(area.x, area.y + 3 * row_h))
            return
        total = sum(sums) + float(self.filters.bias[d["k"]])
        out = float(d["out"][oi, oj])
        tx = area.x + r.u(470)
        y = area.y + r.u(20)
        for s, color in ((t("x1_total", b=f"{self.filters.bias[d['k']]:+.1f}", v=f"{total:+.1f}"), WHITE),
                         (t("x1_silu", v=f"{out:+.1f}"), YELLOW)):
            for line in r.wrap(s, 18, area.right - tx):
                r.text(line, 18, color, topleft=(tx, y))
                y += r.u(25)
            y += r.u(14)
        r.text(t("x1_big"), 15, GREY, bold=False, topleft=(area.x, area.y + 3 * row_h))

    def _draw_full(self, r, area, d, step):
        size = r.u(190)
        r.screen.blit(surface(self._photo, size), area.topleft)
        r.text(t("x1_photo"), 15, GREY, bold=False, topleft=(area.x, area.y + size + r.u(6)))
        fx = area.x + size + r.u(16)
        full = d["full"]
        r.screen.blit(surface(heat_rgb(full), size), (fx, area.y))
        r.text(t("x1_full"), 15, GREY, bold=False, topleft=(fx, area.y + size + r.u(6)))
        ny = area.y + size + r.u(34)
        for line in r.wrap(t("x1_border"), 15, size * 2 + r.u(16), bold=False):
            r.text(line, 15, GREY, bold=False, topleft=(area.x, ny))
            ny += r.u(20)
        if step < 4:
            return
        tile = r.u(90)
        tx = fx + size + r.u(24)
        for i, m in enumerate(d["others"]):
            r.screen.blit(surface(heat_rgb(m), tile),
                          (tx + (i % 2) * (tile + r.u(10)), area.y + (i // 2) * (tile + r.u(10))))
        r.text(t("x1_others", k=len(d["others"]), n=d["n"]), 15, GREY, bold=False,
               topleft=(tx, area.y + size + r.u(6)))


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])   # the classic edge filter: right minus left
SIMPLE_WIDTH = 320       # the camera image is made this wide before the filters run


class SimpleFilter:
    """The simple version of this page: the live camera image next to its edges, found with a
    classic edge filter that people made (Sobel: right minus left, and a second one for bottom
    minus top; white = big difference). Then one filter that the model learned itself, live on the
    same image: of the first-layer filters that react to edges, the one with the most contrast
    when the page opens. Live, so the audience can wave and see the lines follow."""

    def __init__(self):
        self.filters = shared_first_layer()
        self.k = None
        self._frame = None
        self.small = self.edges = self.learned = None

    def preload(self):
        self.filters.preload()

    def open(self, now):
        self.k, self._frame = None, None

    def update(self, frame):
        if frame is None or frame is self._frame:
            return
        self._frame = frame
        fh, fw = frame.shape[:2]
        small = cv2.resize(frame, (SIMPLE_WIDTH, round(SIMPLE_WIDTH * fh / fw)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        self.edges = np.clip(mag / max(1.0, float(np.percentile(mag, 99))) * 255, 0, 255).astype(np.uint8)
        self.small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        if self.filters.ready:
            f, x = self.filters, self.small / 255.0
            if self.k is None:
                self.k = best_edge_filter(f, first_layer(x, f.weights, f.bias))
            self.learned = edge_strength_rgb(first_layer(x, f.weights[self.k:self.k + 1],
                                                         f.bias[self.k:self.k + 1])[0])

    def draw(self, r, now):
        sw, sh = r.size
        r.shade(pygame.Rect(0, 0, sw, sh), 235)
        _, step = draw_steps(r, t("x1s_title"), t("x1s_steps"), r.u(40), r.top + r.u(24), r.u(500), size=23)
        r.text(t("x3_close"), 18, GREY, bottomleft=(r.u(40), sh - r.u(20)))
        if self.small is None:
            r.text(t("x1_wait"), 26, YELLOW, center=(sw - r.u(360), sh // 2))
            return
        x0, y0 = sw - r.u(700), r.top + r.u(30)
        w = r.u(330)
        h = round(w * self.small.shape[0] / self.small.shape[1])
        cam = pygame.Rect(x0, y0, w, h)
        blit_image(r, self.small, cam)
        r.text(t("x1s_camera"), 16, GREY, bold=False, topleft=(cam.x, cam.bottom + r.u(6)))
        if step >= 1:
            edge = pygame.Rect(cam.right + r.u(30), y0, w, h)
            blit_image(r, self.edges, edge)
            pygame.draw.rect(r.screen, WHITE, edge.inflate(4, 4), 1)
            r.text("→", 34, WHITE, center=((cam.right + edge.x) // 2, cam.centery))
            r.text(t("x1s_edges"), 16, GREY, bold=False, topleft=(edge.x, edge.bottom + r.u(6)))
        y = cam.bottom + r.u(50)
        if step >= 3:   # the filter as a small grid of numbers
            c = r.u(40)
            for i in range(3):
                for j in range(3):
                    v = SOBEL_X[i, j]
                    rect = pygame.Rect(x0 + j * c, y + i * c, c, c)
                    if v:
                        pygame.draw.rect(r.screen, PLUS if v > 0 else MINUS, rect)
                    pygame.draw.rect(r.screen, (120, 120, 120), rect, 1)
                    r.text(f"{v:+d}" if v else "0", 18, WHITE, center=rect.center)
            ty = y + 3 * c + r.u(8)
            for line in r.wrap(t("x1s_kernel"), 16, r.u(300), bold=False):
                r.text(line, 16, GREY, bold=False, topleft=(x0, ty))
                ty += r.u(21)
        if step >= 4 and self.learned is not None:
            lw = r.u(300)
            lh = round(lw * self.learned.shape[0] / self.learned.shape[1])
            tile = pygame.Rect(sw - r.u(20) - lw, y, lw, lh)
            blit_image(r, self.learned, tile)
            pygame.draw.rect(r.screen, WHITE, tile.inflate(4, 4), 1)
            ty = tile.bottom + r.u(6)
            for line in r.wrap(t("x1s_learned"), 16, lw, bold=False):
                r.text(line, 16, GREY, bold=False, topleft=(tile.x, ty))
                ty += r.u(21)
