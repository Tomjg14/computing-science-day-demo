"""Shared round logic of the detection games: wait -> countdown -> playing -> result,
plus the team name and the leaderboard entry after a round."""
import math

import pygame

from .. import config
from ..draw import WHITE, YELLOW
from ..texts import t

GREEN = (60, 220, 120)
ORANGE = (255, 140, 40)
CYAN = (0, 220, 255)


def conf_color(c):
    return GREEN if c >= config.CONF_SURE else YELLOW if c >= config.CONF_UNSURE else ORANGE


class RoundGame:
    game_id = ""
    name_key = ""
    label = ""            # short English name for the operator panel
    round_time = 20.0

    def __init__(self, app=None):
        self.phase, self.phase_start = "wait", 0.0
        self.result = None    # {"score", "detail", "record"} of the last round, until it is saved
        self.instructions = True  # the "how to play" card shows until the first round starts

    # --- hooks for each game ---
    def start_round(self, now):
        """Reset the round values."""

    def observe(self, app, det, now, dt):
        """Read the detections every update, in every phase."""

    def play(self, app, det, now, dt):
        """Game logic while the round runs."""

    def score(self):
        """(score, detail) at the end of a round."""
        return 0, ""

    def extra_key(self, key, app):
        return False

    # --- shared logic ---
    def action_label(self):
        return f"Start round ({self.label})" if self.phase in ("wait", "result") else None

    def can_name(self):
        return self.phase in ("wait", "result") and self.result is not None and self.result["score"] > 0

    def on_key(self, key, app):
        if key == pygame.K_i:
            self.instructions = not self.instructions
            return True
        if key == pygame.K_SPACE and self.phase in ("wait", "result"):
            self.instructions = False
            self.result = None
            self.start_round(app.now)
            self.phase, self.phase_start = "countdown", app.now
            return True
        if key == pygame.K_n and self.can_name():
            app.start_text_input(t("team_prompt"), lambda name: self._save(app, name))
            return True
        return self.extra_key(key, app)

    def _save(self, app, name):
        if not name.strip() or self.result is None:
            return
        entry = app.board.add(self.game_id, name, self.result["score"], self.result["detail"])
        self.result = None
        app.show_board(self.game_id, 5, highlight=entry)

    def update(self, app, det, now, dt):
        self.observe(app, det, now, dt)
        el = now - self.phase_start
        if self.phase == "countdown" and el >= config.COUNTDOWN_TIME:
            self.phase, self.phase_start = "playing", now
        elif self.phase == "playing":
            self.play(app, det, now, dt)
            if el >= self.round_time:
                score, detail = self.score()
                record = app.board.best(self.game_id)
                self.result = {"score": score, "detail": detail,
                               "record": score > 0 and (record is None or score > record["score"])}
                self.phase, self.phase_start = "result", now

    # --- shared drawing ---
    def draw_instructions(self, r):
        if self.instructions:
            prefix = self.name_key.replace("_name", "")
            r.instructions(t("instr_title") + ": " + t(self.name_key), t(prefix + "_steps"), t("instr_why"),
                           t(prefix + "_why"), (r.u(24), r.top + r.u(76)))

    def draw_task(self, r, text):
        sw, _ = r.size
        r.banner(r.top, r.u(60), 120)
        r.text(text, 36, YELLOW, center=(sw // 2, r.top + r.u(30)))

    def draw_phase(self, r, app, now):
        """Countdown, timer and result card. Returns True when the card covers the game."""
        sw, sh = r.size
        el = now - self.phase_start
        if self.phase == "countdown":
            n = math.ceil(config.COUNTDOWN_TIME - el)
            r.text(t("get_ready"), 44, WHITE, center=(sw // 2, sh // 2 - r.u(90)))
            r.text(str(n), 200, YELLOW, center=(sw // 2, sh // 2 + r.u(40)))
            return True
        if self.phase == "playing":
            left = max(0.0, self.round_time - el)
            r.text(f"{left:4.1f}", 72, YELLOW if left > 5 else (240, 70, 70), topright=(sw - r.u(30), r.top + r.u(70)))
        if self.phase == "result" and self.result is not None:
            if app.text_input is None:  # while the team name is typed, only the name card shows
                self.draw_result(r, sw, sh)
            return True
        return False

    def result_lines(self):
        """(main line, line when the score is zero)."""
        return "", ""

    def draw_result(self, r, sw, sh):
        res = self.result
        main, zero = self.result_lines()
        card = pygame.Rect(sw // 2 - r.u(400), sh // 2 - r.u(150), r.u(800), r.u(300))
        r.shade(card, 215, r.u(18))
        r.text(t("round_over"), 56, WHITE, midtop=(card.centerx, card.y + r.u(20)))
        if res["score"] > 0:
            r.text(main, 38, YELLOW, midtop=(card.centerx, card.y + r.u(100)))
            if res["record"]:
                r.text(t("new_record"), 30, GREEN, midtop=(card.centerx, card.y + r.u(160)))
            r.text(t("team_ask"), 24, WHITE, shadow=False, midtop=(card.centerx, card.y + r.u(230)))
        else:
            r.text(zero, 30, YELLOW, midtop=(card.centerx, card.y + r.u(120)))
