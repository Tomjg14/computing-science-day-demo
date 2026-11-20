"""Stage 3 - Hide from the AI! The evasion game: body seen, face not seen = invisible.

Two modes (key M): hard uses the modern face model (YOLO, a confidence score), easy uses the
classic Haar cascade (a count of overlapping hits). With the scores off (key V), the bar is
hidden until the round starts, so players do not test their disguise in advance.

Key E shows an explanation: why the old model is easier to fool than the modern one. On each
player's face it draws two classic Haar tests and checks them on the real camera image.
"""
import math
import random

import cv2
import pygame

from .. import config
from ..draw import CYAN, GREEN, PLAYER_COLORS, PURPLE, RED, WHITE, YELLOW, mix
from ..texts import t
from .base import Stage, draw_faces

# Two classic Haar tests (Viola & Jones, 2001), as parts of the face box: (x1, y1, x2, y2).
# Each test asks: are the dark parts darker than the light parts?
HAAR_TESTS = [
    ("x3_test_eyes", [(0.12, 0.28, 0.88, 0.46)], [(0.12, 0.46, 0.88, 0.64)]),            # eyes / cheeks
    ("x3_test_nose", [(0.16, 0.30, 0.40, 0.46), (0.60, 0.30, 0.84, 0.46)], [(0.40, 0.30, 0.60, 0.46)]),
]


def _part(box, f):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return x1 + f[0] * w, y1 + f[1] * h, x1 + f[2] * w, y1 + f[3] * h


def _mean(gray, b):
    crop = gray[max(0, int(b[1])):max(0, int(b[3])), max(0, int(b[0])):max(0, int(b[2]))]
    return float(crop.mean()) if crop.size else None


def haar_test(gray, face_box, dark, light):
    """True when the dark parts are darker than the light parts (mean gray value), None if unknown."""
    d = [_mean(gray, _part(face_box, f)) for f in dark]
    li = [_mean(gray, _part(face_box, f)) for f in light]
    if None in d or None in li:
        return None
    return sum(d) / len(d) < sum(li) / len(li)


def _fill(r, rect, color, alpha):
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    s.fill((*color, alpha))
    r.screen.blit(s, rect.topleft)


class EvasionStage(Stage):
    name = "3 evasion game"
    title_key = "s3_title"
    sub_key = "s3_sub"

    def enter(self, app):
        self.phase = "wait"          # wait | countdown | playing | result
        self.phase_start = app.now
        self.particles = []          # confetti: [x, y, vx, vy, color, life] in screen pixels
        self.instructions = True     # the "how to play" card shows until the first round starts
        self.explain = False         # the model explanation (key E)

    def _set(self, phase, now):
        self.phase, self.phase_start = phase, now

    def action_label(self, app):
        return "Start round" if self.phase in ("wait", "result") else None

    def on_key(self, key, app):
        if key == pygame.K_i:
            self.instructions = not self.instructions
            self.explain = self.explain and not self.instructions
            return True
        if key == pygame.K_e:
            self.explain = not self.explain
            self.instructions = self.instructions and not self.explain
            return True
        if key == pygame.K_SPACE and self.phase in ("wait", "result"):
            self.instructions = self.explain = False
            for p in app.players.players.values():
                p.streak, p.won, p.won_at = 0.0, False, None
            self._set("countdown", app.now)
            return True
        return False

    def update(self, app, det, now, dt):
        el = now - self.phase_start
        if self.phase == "countdown" and el >= config.COUNTDOWN_TIME:
            self._set("playing", now)
        elif self.phase == "playing":
            for p in app.players.players.values():
                if p.invisible:
                    p.streak += dt
                    if p.streak >= config.WIN_STREAK:  # a win; a new streak can win again
                        p.won, p.won_at, p.streak = True, now, 0.0
                        app.stats["invisible"] += 1
                        self._confetti(app.renderer, p)
                else:
                    p.streak = 0.0
            if el >= config.ROUND_TIME:
                self._set("result", now)
        elif self.phase == "result" and el >= config.RESULT_TIME:
            self._set("wait", now)

    def animate(self, app, now, dt):
        for pt in self.particles:
            pt[0] += pt[2] * dt * 60
            pt[1] += pt[3] * dt * 60
            pt[3] += 0.5 * dt * 60
            pt[5] -= dt
        self.particles = [pt for pt in self.particles if pt[5] > 0]

    def _confetti(self, r, p):
        box = r.rect(p.body)
        for _ in range(90):
            self.particles.append([box.centerx, box.y + r.u(40), random.uniform(-9, 9), random.uniform(-14, -2),
                                   random.choice(PLAYER_COLORS + [YELLOW]), 1.6])

    @staticmethod
    def _draw_bar(r, p, box, easy):
        """How strongly the face detector sees this face, with the "seen" limit as a white line.
        Hard mode: YOLO confidence 0-100%. Easy mode: Haar hits (no percentage: hits are a count)."""
        if easy:
            frac = min(1.0, p.haar_hits / config.HAAR_HITS_FULL)
            limit = config.HAAR_MIN_HITS / config.HAAR_HITS_FULL
            label = t("face_hits", n=p.haar_hits)
        else:
            frac, limit, label = p.face_conf, config.FACE_VISIBLE_CONF, f"{t('face')} {p.face_conf:.0%}"
        bw = max(r.u(160), box.w - r.u(40))
        bar = pygame.Rect(box.centerx - bw // 2, max(r.u(115), box.y - r.u(44)), bw, r.u(26))
        r.bar(bar, frac, mix(RED, GREEN, frac))
        tx = bar.x + int(bar.w * limit)
        pygame.draw.line(r.screen, WHITE, (tx, bar.y - r.u(6)), (tx, bar.bottom + r.u(6)), r.u(3))
        r.text(label, 16, WHITE, shadow=False, midleft=(bar.x + r.u(8), bar.centery))

    def draw(self, r, app, det, now):
        sw, sh = r.size
        el = now - self.phase_start
        players = app.players.players
        easy = config.EVASION_MODE == "easy"
        scores = config.SHOW_SCORES or self.phase == "playing"
        if not easy:  # in easy mode the Haar cascade plays, so no YOLO boxes
            draw_faces(r, det, skip=app.players.player_faces(), scores=scores)

        for p in players.values():
            if p.body is None:
                continue
            box = r.rect(p.body)
            if p.invisible and self.phase == "playing":
                pulse = r.u(6) + int(r.u(4) * math.sin(now * 10))
                pygame.draw.rect(r.screen, PURPLE, box, pulse, border_radius=r.u(14))
                r.text(t("invisible"), 46, PURPLE, center=(box.centerx, box.y + r.u(60)))
                w = r.u(240)
                r.bar(pygame.Rect(box.centerx - w // 2, box.y + r.u(100), w, r.u(18)),
                      p.streak / config.WIN_STREAK, PURPLE)
            else:
                pygame.draw.rect(r.screen, p.color, box, r.u(4), border_radius=r.u(14))
            if p.won_at is not None and now - p.won_at < config.WIN_SHOW_TIME:
                r.text(t("won"), 46, YELLOW, center=(box.centerx, box.y + r.u(160)))
            if scores:
                self._draw_bar(r, p, box, easy)

        for pt in self.particles:
            pygame.draw.rect(r.screen, pt[4], (int(pt[0]), int(pt[1]), r.u(10), r.u(14)))

        r.banner(sh - r.u(56), r.u(56))
        r.text(t("invisible_today", n=app.stats["invisible"]), 26, YELLOW, center=(sw // 2, sh - r.u(28)))
        r.text(t("mode_easy" if easy else "mode_hard"), 18, WHITE, shadow=False, bold=False,
               midleft=(r.u(20), sh - r.u(28)))

        if not players and self.phase != "result":
            r.text(t("no_players"), 44, WHITE, center=(sw // 2, sh // 2))
        if self.phase == "countdown":
            n = math.ceil(config.COUNTDOWN_TIME - el)
            r.text(t("get_ready"), 44, WHITE, center=(sw // 2, sh // 2 - r.u(100)))
            r.text(str(n), 200, YELLOW, center=(sw // 2, sh // 2 + r.u(30)))
        elif self.phase == "playing":
            left = max(0.0, config.ROUND_TIME - el)
            r.text(f"{left:4.1f}", 72, YELLOW if left > 5 else RED, topright=(sw - r.u(30), r.u(120)))
        elif self.phase == "result":
            winners = sum(p.won for p in players.values())
            r.banner(sh // 2 - r.u(90), r.u(180), 200)
            r.text(t("round_over"), 72, WHITE, center=(sw // 2, sh // 2 - r.u(30)))
            msg = t("winners", n=winners) if winners else t("no_winners")
            r.text(msg, 30, YELLOW, center=(sw // 2, sh // 2 + r.u(50)))
        elif self.phase == "wait" and players:
            r.text(t("next_round"), 30, WHITE, center=(sw // 2, sh - r.u(100)))
        if self.instructions:
            r.instructions(t("instr_title") + ": " + t("s3_title"), t("evasion_steps"), t("instr_why"),
                           t("evasion_why"), (r.u(24), r.u(125)))
        if self.explain:
            self._draw_haar_tests(r, app, players)
            self._draw_explain(r)

    @staticmethod
    def _draw_haar_tests(r, app, players):
        """Two classic Haar tests on each player's face, checked on the real camera image."""
        if app.frame is None:
            return
        gray = cv2.cvtColor(app.frame, cv2.COLOR_BGR2GRAY)
        active = int(app.now / 2.5) % len(HAAR_TESTS)  # the tests overlap: show one at a time
        for p in players.values():
            if p.face is None:
                continue
            fbox = r.rect(p.face.box)
            y = fbox.bottom + r.u(8)
            for i, (key, dark, light) in enumerate(HAAR_TESTS):
                ok = haar_test(gray, p.face.box, dark, light)
                if i == active:
                    for f in dark:
                        _fill(r, r.rect(_part(p.face.box, f)), (0, 0, 0), 120)
                    for f in light:
                        _fill(r, r.rect(_part(p.face.box, f)), (255, 255, 255), 120)
                color = (GREEN if ok else RED) if i == active else mix((GREEN if ok else RED), (0, 0, 0), 0.25)
                answer = "?" if ok is None else t("yes") if ok else t("no")
                y = r.text(f"{t(key)} {answer}", 18, color, topleft=(fbox.x, y)).bottom + r.u(2)
            pygame.draw.rect(r.screen, CYAN, fbox, r.u(2), border_radius=r.u(6))

    @staticmethod
    def _draw_explain(r):
        """Card on the left: old AI against modern AI."""
        pad, w = r.u(18), r.u(520)
        inner = w - 2 * pad - r.u(18)
        blocks = [(t("x3_old"), [r.wrap(s, 19, inner, bold=False) for s in t("x3_old_lines")]),
                  (t("x3_new"), [r.wrap(s, 19, inner, bold=False) for s in t("x3_new_lines")])]
        title = r.wrap(t("x3_title"), 26, w - 2 * pad)
        lh = r.u(24)
        h = (pad + len(title) * r.u(32) + r.u(10)
             + sum(r.u(34) + sum(len(x) * lh + r.u(6) for x in lines) + r.u(10) for _, lines in blocks)
             + r.u(30) + pad)
        card = pygame.Rect(r.u(24), r.u(125), w, h)
        r.shade(card, 215, r.u(16))
        pygame.draw.rect(r.screen, CYAN, card, r.u(2), border_radius=r.u(16))
        y = card.y + pad
        for line in title:
            r.text(line, 26, CYAN, shadow=False, topleft=(card.x + pad, y))
            y += r.u(32)
        y += r.u(10)
        for (head, lines), color in zip(blocks, (YELLOW, GREEN)):
            r.text(head, 22, color, shadow=False, topleft=(card.x + pad, y))
            y += r.u(34)
            for item in lines:
                pygame.draw.circle(r.screen, color, (card.x + pad + r.u(5), y + r.u(11)), r.u(4))
                for line in item:
                    r.text(line, 19, WHITE, shadow=False, bold=False, topleft=(card.x + pad + r.u(18), y))
                    y += lh
                y += r.u(6)
            y += r.u(10)
        r.text(t("x3_live"), 17, CYAN, shadow=False, bold=False, topleft=(card.x + pad, y))
