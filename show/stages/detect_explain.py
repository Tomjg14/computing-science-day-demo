"""Stage 2 explanation (key E in free detection): how YOLO finds objects, live, with real numbers.

The camera image is split into squares; every square gives a box (where) and a score for each
of the 80 object types (what), all in one look at the image (You Only Look Once). The page shows
the grid, the best score per square (colored like the stage 2 boxes), one square with the box
that it predicts itself, the top 5 types of that square (detailed only), and the final boxes.
The numbers come from a copy of the object model (explain.ObjectGrid).
Simple (default; detailed with Shift+E): layers of filters, squares, where and what, three square
sizes, one answer per object, learned from photos.
"""
import numpy as np
import pygame

from .. import config
from ..draw import BLACK, CYAN, GREY, WHITE, YELLOW
from ..explain import ObjectGrid
from ..players import rank_objects
from ..texts import object_name, t
from ..vision import Obj
from .page_parts import draw_steps

MIN_SHOWN = 0.10      # squares with a lower best score stay empty


def draw_yolo_note(r, x, y, w):
    """A small yellow card: why the model is called YOLO. Returns its bottom."""
    lines = r.wrap(t("yolo_note"), 18, w - r.u(28), bold=False)
    card = pygame.Rect(x, y, w, r.u(46) + len(lines) * r.u(23))
    r.shade(card, 200, r.u(12))
    pygame.draw.rect(r.screen, YELLOW, card, r.u(2), border_radius=r.u(12))
    r.text(t("yolo_title"), 20, YELLOW, shadow=False, topleft=(x + r.u(14), y + r.u(10)))
    for i, line in enumerate(lines):
        r.text(line, 18, WHITE, shadow=False, bold=False, topleft=(x + r.u(14), y + r.u(38) + i * r.u(23)))
    return card.bottom


class DetectExplainer:
    def __init__(self):
        self.grid = ObjectGrid()
        self.on = False

    def preload(self):
        self.grid.preload()

    def toggle(self, now):
        self.on = not self.on
        self.grid.result = None

    def update(self, app):
        if not self.on or app.frame is None:
            return
        players = list(app.players.players.values())

        def pick(dets):
            objs = [Obj(i, name, box, conf) for i, (name, conf, box) in enumerate(dets)]
            ranked = rank_objects(objs, players)     # held first, then the largest; never furniture
            return ranked[0].id if ranked else None
        self.grid.submit(app.frame, pick)

    def draw(self, r, app, now):
        from .detect import conf_color
        sw, sh = r.size
        r.shade(pygame.Rect(0, 0, sw, sh), 235)
        x, y, w = r.u(40), r.top + r.u(24), r.u(525)
        detail = config.EXPLAIN_DETAIL
        steps = t("x2_steps" if detail else "x2s_steps")
        y, step = draw_steps(r, t("x2_title"), steps, x, y, w, size=21)
        if detail and step == len(steps) - 1:
            draw_yolo_note(r, x, y + r.u(8), w)
        r.text(t("x3_close"), 18, GREY, bottomleft=(x, sh - r.u(20)))

        area = pygame.Rect(sw - r.u(680), r.top + r.u(24), r.u(650), sh - r.top - r.u(48))
        res = self.grid.result
        if res is None:
            r.text(t("x2_wait"), 26, YELLOW, center=area.center)
            return
        img = res["image"]
        ih, iw = img.shape[:2]
        scale = area.w / iw
        vw, vh = area.w, int(ih * scale)
        surf = pygame.image.frombuffer(np.ascontiguousarray(img).tobytes(), (iw, ih), "RGB")
        r.screen.blit(pygame.transform.smoothscale(surf, (vw, vh)), area.topleft)
        fw, fh = res["frame_size"]
        to_screen = lambda px, py: (area.x + px / fw * vw, area.y + py / fh * vh)  # noqa: E731

        # from which step each part shows (step = index of the newest step; -1 = none yet)
        if detail:
            grid_from, heat_from, square_from, sizes_from, boxes_from = -1, 1, 2, 99, 3
        else:
            grid_from, heat_from, square_from, sizes_from, boxes_from = 1, 2, 2, 3, 4

        heat = res["heat"]
        gh, gw = heat.shape
        cw, ch = vw / gw, vh / gh
        if step >= heat_from:   # the best score of each square, colored like the boxes of stage 2
            cell = pygame.Surface((int(cw) + 1, int(ch) + 1), pygame.SRCALPHA)
            for i in range(gh):
                for j in range(gw):
                    p = heat[i, j]
                    if p >= MIN_SHOWN:
                        cell.fill((*conf_color(p), int(60 + 150 * min(1.0, p))))
                        r.screen.blit(cell, (area.x + j * cw, area.y + i * ch))
        if step >= grid_from:
            for i in range(gh + 1):
                pygame.draw.line(r.screen, WHITE, (area.x, area.y + i * ch), (area.x + vw, area.y + i * ch), 1)
            for j in range(gw + 1):
                pygame.draw.line(r.screen, WHITE, (area.x + j * cw, area.y), (area.x + j * cw, area.y + vh), 1)
        y = area.y + vh + r.u(8)
        if step >= grid_from:
            r.text(t("x2_grid"), 16, GREY, bold=False, topleft=(area.x, y))
        r.text(t("x2_real"), 16, CYAN, bold=False, topright=(area.right, y))
        y += r.u(30)

        if step >= boxes_from:   # the final boxes
            for name, conf, (x1, y1, x2, y2) in res["dets"]:
                a, b = to_screen(x1, y1)
                c, d = to_screen(x2, y2)
                pygame.draw.rect(r.screen, conf_color(conf), (a, b, c - a, d - b), r.u(3), border_radius=r.u(6))
                r.text(f"{object_name(name)} {conf:.0%}", 16, conf_color(conf), bottomleft=(a, b - r.u(2)))

        focus = res["focus"]
        if step >= square_from:
            if focus is None:
                r.text(t("x2_none"), 22, YELLOW, topleft=(area.x, y))
                return
            y = self._draw_square(r, focus, to_screen, area, y)
            if detail:
                r.text(t("x2_top"), 20, YELLOW, topleft=(area.x, y))
                y += r.u(30)
                for name, p in focus["top"]:
                    r.text(object_name(name), 18, WHITE, topleft=(area.x, y))
                    bar = pygame.Rect(area.x + r.u(200), y + r.u(2), r.u(300), r.u(18))
                    r.bar(bar, p, conf_color(p))
                    r.text(f"{p:.1%}", 18, WHITE, topleft=(bar.right + r.u(14), y))
                    y += r.u(28)
        if step >= sizes_from:
            self._draw_sizes(r, area, y, vw / fw)

    def _draw_square(self, r, focus, to_screen, area, y):
        """One square (yellow) and the box that it predicts itself, with a line between them.
        Returns y below its note."""
        a, b = to_screen(*focus["square"][:2])
        c, d = to_screen(*focus["square"][2:])
        sq = pygame.Rect(round(a), round(b), max(4, round(c - a)), max(4, round(d - b)))
        a, b = to_screen(*focus["box"][:2])
        c, d = to_screen(*focus["box"][2:])
        box = pygame.Rect(round(a), round(b), round(c - a), round(d - b)).clip(area)
        pygame.draw.rect(r.screen, YELLOW, box, r.u(3))
        corner = (box.x, box.y) if sq.centery > box.centery else (box.x, box.bottom)
        pygame.draw.line(r.screen, YELLOW, sq.center, corner, r.u(2))
        pygame.draw.rect(r.screen, BLACK, sq.inflate(r.u(4), r.u(4)), r.u(2))
        pygame.draw.rect(r.screen, YELLOW, sq, r.u(4))
        name, p = focus["top"][0]
        r.text(t("x2_square", name=object_name(name), p=f"{p:.0%}"), 18, YELLOW, topleft=(area.x, y))
        y += r.u(28)
        for line in r.wrap(t("x2_where"), 16, area.w, bold=False):
            r.text(line, 16, GREY, bold=False, topleft=(area.x, y))
            y += r.u(21)
        return y + r.u(10)

    def _draw_sizes(self, r, area, y, screen_per_frame_px):
        """The three square sizes of the model (8, 16 and 32 pixels of its input), at screen scale."""
        x = area.x
        r.text(t("x2_sizes"), 18, WHITE, topleft=(x, y))
        y += r.u(30)
        frame_per_input_px = self.grid.result["frame_size"][0] / self.grid.WIDTH
        for k, s in enumerate((8, 16, 32)):
            side = max(3, round(s * frame_per_input_px * screen_per_frame_px))
            pygame.draw.rect(r.screen, YELLOW, (x, y + r.u(40) - side, side, side), r.u(2))
            r.text(t("x2_size_names")[k], 16, GREY, bold=False, topleft=(x, y + r.u(46)))
            x += max(side, r.u(90)) + r.u(30)
