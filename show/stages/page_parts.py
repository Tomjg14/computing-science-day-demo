"""Drawing parts that the explanation pages share."""
import cv2
import numpy as np
import pygame

from ..draw import BLACK, CYAN, GREEN, GREY, RED, WHITE, YELLOW, mix
from ..texts import t

SCAN_TIME = 6.0        # seconds for one slow sweep of the window over the image (illustration)


class Steps:
    """The text steps of the open explanation page. Nothing moves on by itself: the presenter
    shows the next step (Page Down, N, or a presentation clicker) or goes one step back (Page Up).
    A page opens with its title and image only. One object for all pages: only one page is open
    at a time, and the app starts it again when another page opens."""

    def __init__(self):
        self.total = 0       # steps on the open page (set when the page draws them)
        self.times = []      # when each shown step appeared

    def reset(self):
        self.total, self.times = 0, []

    @property
    def shown(self):
        return len(self.times)

    @property
    def newest(self):
        """Index of the newest step on screen; -1 = none yet."""
        return len(self.times) - 1

    def next(self, now):
        if self.shown < self.total:
            self.times.append(now)
            return True
        return False

    def back(self):
        if self.times:
            self.times.pop()
            return True
        return False

    def age(self, i, now):
        """Seconds since step i appeared; 0 when it is not on screen."""
        return now - self.times[i] if i < len(self.times) else 0.0


STEPS = Steps()


def draw_steps(r, title, steps, x, y, w, size=22):
    """The title and the numbered steps that the presenter has shown so far (STEPS), and small
    dots at the bottom that show how many steps there are. Returns (y below the last step,
    index of the newest step; -1 = none)."""
    STEPS.total = len(steps)
    r.text(title, 30, CYAN, topleft=(x, y))
    y += r.u(50)
    shown = min(len(steps), STEPS.shown)
    dot_y = r.size[1] - r.u(31)            # on the line of the "E: ..." hint, right of it
    for i in range(len(steps)):
        center = (x + r.u(400) + i * r.u(22), dot_y)
        if i < shown:
            pygame.draw.circle(r.screen, YELLOW, center, r.u(6))
        else:
            pygame.draw.circle(r.screen, GREY, center, r.u(6), 2)
    for i in range(shown):
        circle = (x + r.u(15), y + r.u(15))
        pygame.draw.circle(r.screen, YELLOW, circle, r.u(15))
        r.text(str(i + 1), 20, BLACK, shadow=False, center=circle)
        for line in r.wrap(steps[i], size, w - r.u(44)):
            r.text(line, size, WHITE, topleft=(x + r.u(44), y))
            y += r.u(size + 6)
        y += r.u(10)
    return y, shown - 1


def blit_image(r, img, rect):
    """An RGB or gray image (numpy) scaled into rect."""
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    img = cv2.resize(np.ascontiguousarray(img), (rect.w, rect.h), interpolation=cv2.INTER_AREA)
    r.screen.blit(pygame.image.frombuffer(np.ascontiguousarray(img).tobytes(), (rect.w, rect.h), "RGB"), rect.topleft)


def draw_scan(r, rect, gray, found, now):
    """The old AI's search: the gray camera image with a window that slides over every place in
    three sizes (slowed down: an illustration of what the detector does very fast), and the real
    windows where the Haar cascade finds a face (green).
    gray: small gray camera image; found: boxes (x, y, w, h) in its pixels."""
    blit_image(r, gray, rect)
    sx, sy = rect.w / gray.shape[1], rect.h / gray.shape[0]
    sizes = (0.45, 0.32, 0.22)
    k = (now % (SCAN_TIME * len(sizes))) / SCAN_TIME
    side = int(rect.h * sizes[int(k)])
    stride = max(4, side // 3)
    cols = max(1, (rect.w - side) // stride + 1)
    rows = max(1, (rect.h - side) // stride + 1)
    i = int((k % 1) * cols * rows)
    row, col = divmod(i, cols)
    win = pygame.Rect(rect.x + col * stride, rect.y + row * stride, side, side)
    for x, y, w, h in found:
        pygame.draw.rect(r.screen, GREEN, (rect.x + x * sx, rect.y + y * sy, w * sx, h * sy), r.u(3))
    pygame.draw.rect(r.screen, YELLOW, win, r.u(3))
    pygame.draw.rect(r.screen, WHITE, rect.inflate(4, 4), 1)
    y = rect.bottom + r.u(6)
    for line in r.wrap(t("scan_note"), 15, rect.w, bold=False):
        r.text(line, 15, GREY, bold=False, topleft=(rect.x, y))
        y += r.u(19)


def score_bar(r, rect, score, limit):
    """The face score of the explanation face, colored like the bars of the hiding game, with the
    "seen" limit as a white line. score None = not found."""
    if score is None:
        r.bar(rect, 0.0, RED)
        label, color = t("xf_lost"), RED
    else:
        r.bar(rect, score, mix(RED, GREEN, score))
        label, color = f"{score:.0%}", WHITE
    lx = rect.x + int(rect.w * limit)
    pygame.draw.line(r.screen, WHITE, (lx, rect.y - r.u(6)), (lx, rect.bottom + r.u(6)), r.u(3))
    r.text(label, 20, color, shadow=False, midleft=(rect.x + r.u(10), rect.centery))

