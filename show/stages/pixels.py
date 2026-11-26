"""Stage 1 - What does the computer see? A still of one face turns into color pixels,
and a magnifier shows that each pixel is three numbers: red, green and blue.

Key E: what the AI does with these numbers (filter_page.py). Simple (default): the live camera
image and its edges. Detailed (Shift+E): one real filter of the model on the pixels of the photo.

The still lives in memory only and is dropped when the stage is left (ADR 0001).
"""
import cv2
import numpy as np
import pygame

from .. import config
from ..draw import GREY, WHITE, YELLOW
from ..texts import t
from ..vision import crop_square, largest_face
from .base import Stage, draw_faces
from .filter_page import FilterExplainer, SimpleFilter

PHOTO_END = 1.5        # seconds: the plain photo
PIXEL_END = 4.5        # the photo gets coarser until it reaches the grid size
SPOT_TIME = 1.8        # seconds per magnified pixel
N_SPOTS = 6
CHANNEL_COLORS = [(255, 70, 70), (70, 230, 90), (80, 140, 255)]


def pick_spots(cells, n):
    """Cells with very different colors (farthest-point sampling), starting at the center."""
    g = cells.shape[0]
    flat = cells.reshape(-1, 3).astype(float)
    chosen = [(g // 2) * g + g // 2]
    dist = np.linalg.norm(flat - flat[chosen[0]], axis=1)
    for _ in range(n - 1):
        nxt = int(dist.argmax())
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(flat - flat[nxt], axis=1))
    return [divmod(i, g) for i in chosen]  # (row, col)


def to_surface(rgb):
    rgb = np.ascontiguousarray(rgb)
    return pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB").copy()


class PixelStage(Stage):
    name = "pixel grid"
    title_key = "s1_title"
    sub_key = "s1_sub"

    def enter(self, app):
        self.photo = None
        self.t0 = 0.0
        if self.explainer is None:
            PixelStage.explainer = FilterExplainer()
            PixelStage.simple = SimpleFilter()
        self.explainer.preload()
        self.explainer.close()
        self.explaining = False

    explainer = None       # the detailed filter explanation (key E); made once, it loads the model
    simple = None          # the simple one

    @property
    def explain(self):
        """The explanation page on screen: "filter" or None (for the operator panel)."""
        return "filter" if self.explaining else None

    def _explain(self, on, now):
        self.explaining = on
        if on:
            self.simple.open(now)
            if self.photo is not None:
                self.explainer.open(now, self.photo, self.cells)
        else:
            self.explainer.close()

    def action_label(self, app):
        return "Take photo" if self.photo is None else "New photo"

    def on_key(self, key, app):
        if key == pygame.K_e:
            self._explain(not self.explaining, app.now)
            return True
        if key == pygame.K_SPACE:
            self._explain(False, app.now)
            box = app.face_lock.box if app.face_lock.conf is not None else None   # the explanation face
            if box is None and app.det:
                face = largest_face(app.det.faces, config.FACE_VISIBLE_CONF)
                box = face.box if face is not None else None
            if box is not None:
                crop = crop_square(app.frame, box)
                if crop.size:
                    g = config.PIXEL_GRID
                    self.photo = cv2.cvtColor(cv2.resize(crop, (480, 480)), cv2.COLOR_BGR2RGB)
                    self.cells = cv2.resize(self.photo, (g, g), interpolation=cv2.INTER_AREA)
                    self.spots = pick_spots(self.cells, N_SPOTS)
                    self.channels = None
                    self.t0 = app.now
            return True
        if key == pygame.K_BACKSPACE:
            self.photo = None
            self._explain(False, app.now)
            return True
        return False

    def draw(self, r, app, det, now):
        sw, sh = r.size
        if self.explain and not config.EXPLAIN_DETAIL:
            self.simple.draw(r, now)
            return
        if self.explain and self.explainer.on:
            self.explainer.draw(r, now)
            return
        if self.explain:   # the detailed page needs the photo
            r.shade(pygame.Rect(0, 0, sw, sh), 235)
            r.text(t("x1_need_photo"), 30, YELLOW, center=(sw // 2, sh // 2))
            return
        if self.photo is None:
            draw_faces(r, det)
            if app.overlays:
                r.banner(sh - r.u(90), r.u(90))
                r.text(t("s1_prompt"), 32, YELLOW, center=(sw // 2, sh - r.u(45)))
            return

        el = now - self.t0
        g = config.PIXEL_GRID
        r.shade(pygame.Rect(0, 0, sw, sh), 200)
        size = int(min(sh * 0.70, sw * 0.45))
        size -= size % g                   # whole cells only
        cell = size // g
        px, py = r.u(60), r.u(125)

        # 1) photo -> 2) coarser color blocks
        if el < PHOTO_END:
            img = cv2.resize(self.photo, (size, size), interpolation=cv2.INTER_LINEAR)
        else:
            k = min(1.0, (el - PHOTO_END) / (PIXEL_END - PHOTO_END))
            n = max(g, int(round(160 * (g / 160) ** k)))   # smooth zoom from 160 to g cells
            small = cv2.resize(self.photo, (n, n), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (size, size), interpolation=cv2.INTER_NEAREST)
        r.screen.blit(to_surface(img), (px, py))
        if el >= PIXEL_END:
            for i in range(g + 1):
                pygame.draw.line(r.screen, (30, 30, 30), (px + i * cell, py), (px + i * cell, py + size))
                pygame.draw.line(r.screen, (30, 30, 30), (px, py + i * cell), (px + size, py + i * cell))
        pygame.draw.rect(r.screen, WHITE, (px - 3, py - 3, size + 6, size + 6), 3)

        # explanation on the right, one line per phase
        tx, ty = px + size + r.u(45), py + r.u(10)
        steps = t("s1_steps")
        shown = 1 + (el >= PHOTO_END) + (el >= PIXEL_END)
        for i in range(shown):
            r.text(steps[i], 28, WHITE, topleft=(tx, ty + i * r.u(48)))
        if el < PIXEL_END:
            return

        # 3) magnifier: each pixel is three numbers
        lens_el = el - PIXEL_END
        row, col = self.spots[int(lens_el / SPOT_TIME) % len(self.spots)]
        rv, gv, bv = (int(v) for v in self.cells[row, col])
        cell_rect = pygame.Rect(px + col * cell, py + row * cell, cell, cell)
        pulse = 3 + int(2 * (1 + np.sin(now * 8)))
        pygame.draw.rect(r.screen, YELLOW, cell_rect.inflate(6, 6), pulse)

        box = pygame.Rect(tx, ty + 3 * r.u(48) + r.u(6), r.u(470), r.u(128))
        r.shade(box, 200, r.u(12))
        pygame.draw.line(r.screen, YELLOW, cell_rect.center, (box.x, box.centery), 2)
        swatch = pygame.Rect(box.x + r.u(14), box.y + r.u(14), r.u(100), r.u(100))
        pygame.draw.rect(r.screen, (rv, gv, bv), swatch)
        pygame.draw.rect(r.screen, WHITE, swatch, 2)
        bx, bw = swatch.right + r.u(20), r.u(250)
        for i, (label, v) in enumerate(zip("RGB", (rv, gv, bv))):
            y = box.y + r.u(18) + i * r.u(34)
            r.text(f"{label} {v:3d}", 22, CHANNEL_COLORS[i], shadow=False, mono=True, topleft=(bx, y))
            r.bar(pygame.Rect(bx + r.u(85), y + r.u(4), bw, r.u(18)), v / 255, CHANNEL_COLORS[i])
        r.text(t("s1_scale"), 18, YELLOW, shadow=False, bold=False, topleft=(box.x, box.bottom + r.u(6)))

        # 4) after one round of the magnifier: the three layers
        if lens_el >= SPOT_TIME * 3:
            if self.channels is None:
                th = r.u(110)
                self.channels = []
                for c in range(3):
                    layer = np.zeros_like(self.cells)
                    layer[:, :, c] = self.cells[:, :, c]
                    self.channels.append(to_surface(cv2.resize(layer, (th, th), interpolation=cv2.INTER_NEAREST)))
            ly = box.bottom + r.u(40)
            r.text(t("s1_layers"), 22, WHITE, topleft=(tx, ly))
            for c, surf in enumerate(self.channels):
                x = tx + c * (surf.get_width() + r.u(22))
                r.screen.blit(surf, (x, ly + r.u(34)))
                pygame.draw.rect(r.screen, CHANNEL_COLORS[c], (x - 2, ly + r.u(34) - 2, surf.get_width() + 4,
                                                               surf.get_height() + 4), 2)
        if lens_el >= SPOT_TIME * len(self.spots):
            end_y = box.bottom + r.u(40) + r.u(34) + r.u(110) + r.u(14)
            r.text(t("s1_end"), 30, YELLOW, topleft=(tx, end_y))
            r.text(t("x1_open"), 18, GREY, topleft=(tx, end_y + r.u(44)))

    def update(self, app, det, now, dt):
        if self.explaining:
            self.simple.update(app.frame)
