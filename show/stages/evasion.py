"""Stage 3 - Hide from the AI! The evasion game: body seen, face not seen = invisible.

Two modes (key M): hard uses the modern face model (YOLO, a confidence score), easy uses the
classic Haar cascade (a count of overlapping hits). With the scores off (key V), the bar is
hidden until the round starts, so players do not test their disguise in advance.

Key E shows an explanation, live on the largest face: why the AI can miss a face, with the face
score and measurements of the face area (fail_page.py). E again closes it.
"""
import math
import random

import pygame

from .. import config
from ..draw import GREEN, PLAYER_COLORS, PURPLE, RED, WHITE, YELLOW, mix
from ..texts import t
from .base import Stage, draw_faces
from .fail_page import FailExplainer

class EvasionStage(Stage):
    name = "evasion game"
    title_key = "s3_title"
    sub_key = "s3_sub"

    def enter(self, app):
        self.phase = "wait"          # wait | countdown | playing | result
        self.phase_start = app.now
        self.particles = []          # confetti: [x, y, vx, vy, color, life] in screen pixels
        self.instructions = True     # the "how to play" card shows until the first round starts
        self.extra = 0.0             # seconds added to this round with X
        self.extended_at = None      # time of the last X, for the short "+10 s" on screen
        if self.explainer is None:
            self.explainer = FailExplainer()
        self.explainer.close()

    explainer = None                 # why the AI can miss a face (key E)

    @property
    def explain(self):
        """The explanation page on screen: None or "fail" (for the operator panel)."""
        return "fail" if self.explainer is not None and self.explainer.on else None

    def _set(self, phase, now):
        self.phase, self.phase_start = phase, now

    def action_label(self, app):
        return {"countdown": "Cancel round", "playing": "Stop round now (show result)"}.get(
            self.phase, "Start round")

    def round_running(self, app):
        return self.phase == "playing"

    def time_left(self, now):
        return max(0.0, config.ROUND_TIME + self.extra - (now - self.phase_start))

    def on_key(self, key, app):
        if key == pygame.K_i:
            self.instructions = not self.instructions
            if self.instructions:
                self.explainer.close()
            return True
        if key == pygame.K_e:
            self.explainer.toggle(app.now)   # the page covers the card; the card stays for after it
            return True
        if key == pygame.K_SPACE and self.phase in ("wait", "result"):
            self.instructions = False
            self.explainer.close()
            for p in app.players.players.values():
                p.streak, p.won, p.won_at = 0.0, False, None
            self.extra, self.extended_at = 0.0, None
            self._set("countdown", app.now)
            return True
        if key == pygame.K_SPACE and self.phase == "countdown":   # cancel: nothing was played
            self._set("wait", app.now)
            return True
        if key == pygame.K_SPACE and self.phase == "playing":     # stop now: show the result
            self._set("result", app.now)
            return True
        if key == pygame.K_x and self.phase == "playing":
            self.extra += config.EXTEND_TIME
            self.extended_at = app.now
            return True
        return False

    def update(self, app, det, now, dt):
        self.explainer.update(app, det, now)
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
            if self.time_left(now) <= 0:
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
        if self.explain:
            self.explainer.draw(r, app, now)
            return
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

        if app.overlays:
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
            left = self.time_left(now)
            timer = r.text(f"{left:4.1f}", 72, YELLOW if left > 5 else RED, topright=(sw - r.u(30), r.top + r.u(12)))
            if self.extended_at is not None and now - self.extended_at < 1.5:
                r.text(f"+{config.EXTEND_TIME:.0f} s", 40, GREEN, midtop=(timer.centerx, timer.bottom))
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
                           t("evasion_why"), (r.u(24), r.top + r.u(17)))
