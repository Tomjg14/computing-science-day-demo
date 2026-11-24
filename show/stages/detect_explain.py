"""Stage 2 explanation (key E in free detection): how YOLO finds objects, live, with real numbers.

The camera image is split into squares; every square gives a score for each of the 80 object
types, all in one look at the image (You Only Look Once). The page shows the grid, the best
score per square (colored like the stage 2 boxes), the top 5 types of one object, and the
final boxes. The numbers come from a copy of the object model (explain.ObjectGrid).
"""
import numpy as np
import pygame

from ..draw import BLACK, CYAN, GREY, WHITE, YELLOW
from ..explain import ObjectGrid
from ..players import rank_objects
from ..texts import object_name, t
from ..vision import Obj

STEP_TIME = 4.0
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
        self.start = 0.0

    def preload(self):
        self.grid.preload()

    def toggle(self, now):
        self.on = not self.on
        self.start = now
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
        el = now - self.start
        x, y, w = r.u(40), r.top + r.u(24), r.u(500)
        r.text(t("x2_title"), 32, CYAN, topleft=(x, y))
        y += r.u(54)
        steps = t("x2_steps")
        shown = min(len(steps), 1 + int(el / STEP_TIME))
        for i in range(shown):
            circle = (x + r.u(15), y + r.u(15))
            pygame.draw.circle(r.screen, YELLOW, circle, r.u(15))
            r.text(str(i + 1), 20, BLACK, shadow=False, center=circle)
            for line in r.wrap(steps[i], 22, w - r.u(44)):
                r.text(line, 22, WHITE, topleft=(x + r.u(44), y))
                y += r.u(29)
            y += r.u(12)
        if shown == len(steps):
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

        heat = res["heat"]
        gh, gw = heat.shape
        cw, ch = vw / gw, vh / gh
        step = int(el / STEP_TIME)
        if step >= 1:   # step 2: the best score of each square, colored like the boxes of stage 2
            cell = pygame.Surface((int(cw) + 1, int(ch) + 1), pygame.SRCALPHA)
            for i in range(gh):
                for j in range(gw):
                    p = heat[i, j]
                    if p >= MIN_SHOWN:
                        cell.fill((*conf_color(p), int(60 + 150 * min(1.0, p))))
                        r.screen.blit(cell, (area.x + j * cw, area.y + i * ch))
        for i in range(gh + 1):
            pygame.draw.line(r.screen, WHITE, (area.x, area.y + i * ch), (area.x + vw, area.y + i * ch), 1)
        for j in range(gw + 1):
            pygame.draw.line(r.screen, WHITE, (area.x + j * cw, area.y), (area.x + j * cw, area.y + vh), 1)
        y = area.y + vh + r.u(8)
        r.text(t("x2_grid"), 16, GREY, bold=False, topleft=(area.x, y))
        r.text(t("x2_real"), 16, CYAN, bold=False, topright=(area.right, y))
        y += r.u(30)

        focus = res["focus"]
        if step >= 3:   # step 4: the final boxes
            for name, conf, (x1, y1, x2, y2) in res["dets"]:
                a, b = to_screen(x1, y1)
                c, d = to_screen(x2, y2)
                pygame.draw.rect(r.screen, conf_color(conf), (a, b, c - a, d - b), r.u(3), border_radius=r.u(6))
                r.text(f"{object_name(name)} {conf:.0%}", 16, conf_color(conf), bottomleft=(a, b - r.u(2)))
        if step >= 2:   # step 3: one square and its 5 highest scores
            if focus is None:
                r.text(t("x2_none"), 22, YELLOW, topleft=(area.x, y))
                return
            cx, cy = to_screen(*focus["center"])
            j, i = int((cx - area.x) / cw), int((cy - area.y) / ch)
            pygame.draw.rect(r.screen, YELLOW, (area.x + j * cw, area.y + i * ch, cw, ch), r.u(4))
            r.text(t("x2_top"), 20, YELLOW, topleft=(area.x, y))
            y += r.u(30)
            for name, p in focus["top"]:
                r.text(object_name(name), 18, WHITE, topleft=(area.x, y))
                bar = pygame.Rect(area.x + r.u(200), y + r.u(2), r.u(300), r.u(18))
                r.bar(bar, p, conf_color(p))
                r.text(f"{p:.1%}", 18, WHITE, topleft=(bar.right + r.u(14), y))
                y += r.u(28)
