"""Stage 3 - Hide from the AI! The evasion game: body seen, face not seen = invisible."""
import math
import random

import pygame

from .. import config
from ..draw import GREEN, PLAYER_COLORS, PURPLE, RED, WHITE, YELLOW, mix
from ..texts import t
from .base import Stage, draw_faces


class EvasionStage(Stage):
    name = "3 evasion game"
    title_key = "s3_title"
    sub_key = "s3_sub"

    def enter(self, app):
        self.phase = "wait"          # wait | countdown | playing | result
        self.phase_start = app.now
        self.particles = []          # confetti: [x, y, vx, vy, color, life] in screen pixels
        self.instructions = True     # the "how to play" card shows until the first round starts

    def _set(self, phase, now):
        self.phase, self.phase_start = phase, now

    def action_label(self, app):
        return "Start round" if self.phase in ("wait", "result") else None

    def on_key(self, key, app):
        if key == pygame.K_i:
            self.instructions = not self.instructions
            return True
        if key == pygame.K_SPACE and self.phase in ("wait", "result"):
            self.instructions = False
            for p in app.players.players.values():
                p.streak, p.won = 0.0, False
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
                    if p.streak >= config.WIN_STREAK and not p.won:
                        p.won = True
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

    def draw(self, r, app, det, now):
        sw, sh = r.size
        el = now - self.phase_start
        players = app.players.players
        draw_faces(r, det, skip=app.players.player_faces())

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
            if p.won:
                r.text(t("won"), 46, YELLOW, center=(box.centerx, box.y + r.u(160)))
            # visibility bar: the face confidence score, with the "seen" threshold as a white line
            bw = max(r.u(160), box.w - r.u(40))
            bar = pygame.Rect(box.centerx - bw // 2, max(r.u(115), box.y - r.u(44)), bw, r.u(26))
            r.bar(bar, p.face_conf, mix(RED, GREEN, p.face_conf))
            tx = bar.x + int(bar.w * config.FACE_VISIBLE_CONF)
            pygame.draw.line(r.screen, WHITE, (tx, bar.y - r.u(6)), (tx, bar.bottom + r.u(6)), r.u(3))
            r.text(f"{t('face')} {p.face_conf:.0%}", 16, WHITE, shadow=False, midleft=(bar.x + r.u(8), bar.centery))

        for pt in self.particles:
            pygame.draw.rect(r.screen, pt[4], (int(pt[0]), int(pt[1]), r.u(10), r.u(14)))

        r.banner(sh - r.u(56), r.u(56))
        r.text(t("invisible_today", n=app.stats["invisible"]), 26, YELLOW, center=(sw // 2, sh - r.u(28)))

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
