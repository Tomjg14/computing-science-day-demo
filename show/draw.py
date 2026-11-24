"""Drawing helpers: fonts that scale with the window, text with shadow, banners and labeled boxes."""
import cv2
import pygame

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (150, 150, 150)
GREEN = (60, 220, 120)
RED = (240, 70, 70)
PURPLE = (190, 90, 255)
YELLOW = (255, 210, 60)
CYAN = (0, 220, 255)
PLAYER_COLORS = [(80, 180, 255), (255, 140, 60), (90, 230, 160), (255, 100, 200)]


def class_color(name):
    """A stable, bright color for each object class."""
    hue = (sum(map(ord, name)) * 37) % 360
    c = pygame.Color(0)
    c.hsva = (hue, 75, 100, 100)
    return (c.r, c.g, c.b)


def mix(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Renderer:
    def __init__(self, screen):
        self.screen = screen
        self._fonts = {}
        self.sx = self.sy = 1.0
        self.top = 0  # height of the header line now on screen (0 = no header); games draw below it

    @property
    def size(self):
        return self.screen.get_size()

    def u(self, v):
        """Scale a size designed for a 720 px high window to the real window."""
        return max(1, int(v * self.screen.get_height() / 720))

    def font(self, size, bold=True, mono=False):
        key = (self.u(size), bold, mono)
        if key not in self._fonts:
            self._fonts[key] = pygame.font.SysFont("consolas" if mono else "arial", key[0], bold=bold)
        return self._fonts[key]

    def video(self, frame_bgr):
        """Draw the camera frame over the whole window and remember the scale for boxes."""
        sw, sh = self.size
        fh, fw = frame_bgr.shape[:2]
        self.sx, self.sy = sw / fw, sh / fh
        key = (id(frame_bgr), sw, sh)
        if getattr(self, "_video_key", None) != key:  # convert each camera frame only once
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            surf = pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB")
            self._video = pygame.transform.scale(surf, (sw, sh))
            self._video_key = key
        self.screen.blit(self._video, (0, 0))

    def rect(self, box):
        x1, y1, x2, y2 = box
        return pygame.Rect(int(x1 * self.sx), int(y1 * self.sy), int((x2 - x1) * self.sx), int((y2 - y1) * self.sy))

    def point(self, x, y):
        return int(x * self.sx), int(y * self.sy)

    def text(self, s, size, color=WHITE, shadow=True, bold=True, mono=False, alpha=255, **pos):
        """Draw text; pos is one rect anchor, e.g. center=(x, y) or topleft=(x, y).
        alpha below 255 makes the text see-through (for fading)."""
        f = self.font(size, bold, mono)
        img = f.render(s, True, color)
        img.set_alpha(alpha)
        r = img.get_rect(**pos)
        if shadow:
            off = self.u(min(3, max(1, round(size / 9))))  # small text: a closer shadow, so it reads as an edge
            back = f.render(s, True, BLACK)
            back.set_alpha(alpha)
            self.screen.blit(back, r.move(off, off))
        self.screen.blit(img, r)
        return r

    def shade(self, rect, alpha=160, radius=0):
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(surf, (0, 0, 0, alpha), surf.get_rect(), border_radius=radius)
        self.screen.blit(surf, rect.topleft)

    def banner(self, y, h, alpha=160):
        self.shade(pygame.Rect(0, y, self.size[0], h), alpha)

    def labeled_box(self, box, label, color, width=4, size=26):
        r = self.rect(box)
        pygame.draw.rect(self.screen, color, r, self.u(width), border_radius=self.u(8))
        f = self.font(size)
        img = f.render(label, True, BLACK)
        pad = self.u(6)
        tag = pygame.Rect(r.x, r.y - img.get_height() - 2 * pad, img.get_width() + 2 * pad, img.get_height() + 2 * pad)
        if tag.y < 0:
            tag.y = r.y
        pygame.draw.rect(self.screen, color, tag, border_radius=self.u(6))
        self.screen.blit(img, (tag.x + pad, tag.y + pad))
        return r

    def wrap(self, text, size, width, bold=True):
        f, words, lines, cur = self.font(size, bold), text.split(), [], ""
        for word in words:
            test = f"{cur} {word}".strip()
            if f.size(test)[0] <= width:
                cur = test
            else:
                lines.append(cur)
                cur = word
        return lines + [cur] if cur else lines

    def instructions(self, title, steps, why_label, why, topleft, width=470):
        """A 'how to play' card: title, numbered steps and a 'why' line."""
        pad, w = self.u(18), self.u(width)
        inner = w - 2 * pad - self.u(34)
        step_lines = [self.wrap(s, 21, inner) for s in steps]
        why_lines = self.wrap(why, 19, w - 2 * pad, bold=False)
        lh, wh = self.u(27), self.u(24)
        h = (pad + self.u(46) + sum(len(x) * lh + self.u(8) for x in step_lines) + self.u(34)
             + len(why_lines) * wh + pad)
        card = pygame.Rect(topleft, (w, h))
        self.shade(card, 215, self.u(16))
        pygame.draw.rect(self.screen, YELLOW, card, self.u(2), border_radius=self.u(16))
        self.text(title, 28, YELLOW, shadow=False, topleft=(card.x + pad, card.y + pad))
        y = card.y + pad + self.u(46)
        for i, lines in enumerate(step_lines):
            circle = (card.x + pad + self.u(13), y + self.u(13))
            pygame.draw.circle(self.screen, YELLOW, circle, self.u(13))
            self.text(str(i + 1), 18, BLACK, shadow=False, center=circle)
            for line in lines:
                self.text(line, 21, WHITE, shadow=False, topleft=(card.x + pad + self.u(34), y))
                y += lh
            y += self.u(8)
        y += self.u(8)
        self.text(why_label, 19, CYAN, shadow=False, topleft=(card.x + pad, y))
        y += self.u(26)
        for line in why_lines:
            self.text(line, 19, WHITE, shadow=False, bold=False, topleft=(card.x + pad, y))
            y += wh
        return card

    def bar(self, rect, frac, color, border=WHITE):
        pygame.draw.rect(self.screen, BLACK, rect.inflate(self.u(6), self.u(6)), border_radius=self.u(10))
        inner = rect.copy()
        inner.w = int(rect.w * max(0.0, min(1.0, frac)))
        if inner.w > 0:
            pygame.draw.rect(self.screen, color, inner, border_radius=self.u(8))
        pygame.draw.rect(self.screen, border, rect.inflate(self.u(6), self.u(6)), self.u(2), border_radius=self.u(10))
