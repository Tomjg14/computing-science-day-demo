"""Leaderboards of the detection games.

Stored in a small JSON file so a laptop crash does not wipe a demo day. It holds only
team names (never real names), the game, the score and the time (ADR 0001).
Only entries of today are shown, so every day starts with empty lists.
"""
import json
import time

import pygame

from . import config
from .draw import BLACK, WHITE, YELLOW
from .texts import object_name, t

TOP_N = 5

# game id -> (text key of the game name, how a score is shown)
GAMES = {
    "fool": ("fool_name", lambda e: f"{e['score']:.0%} {object_name(e['detail'])}"),
    "hunt": ("hunt_name", lambda e: t("hunt_board", n=int(e["score"]))),
    "confuse": ("confuse_name", lambda e: t("times", n=int(e["score"]))),
}


def today():
    return time.strftime("%Y-%m-%d")


class Leaderboard:
    def __init__(self, path=None):
        self.path = path or config.LEADERBOARD_FILE
        self.entries = []
        try:
            self.entries = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.entries, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)  # write-then-rename: a crash never leaves a half file

    def add(self, game, team, score, detail=""):
        entry = {"game": game, "team": team.strip()[:24], "score": score, "detail": detail,
                 "date": today(), "time": time.strftime("%H:%M:%S")}
        self.entries.append(entry)
        self._save()
        return entry

    def top(self, game, n=TOP_N):
        rows = [e for e in self.entries if e["game"] == game and e["date"] == today()]
        return sorted(rows, key=lambda e: -e["score"])[:n]

    def best(self, game):
        rows = self.top(game, 1)
        return rows[0] if rows else None

    def remove(self, entry):
        if entry in self.entries:
            self.entries.remove(entry)
            self._save()

    def clear_today(self):
        self.entries = [e for e in self.entries if e["date"] != today()]
        self._save()

    def games_with_entries(self):
        return [g for g in GAMES if self.top(g, 1)]


def board_size(r, board, game, subtitle=False):
    rows = len(board.top(game))
    return r.u(500), r.u(96 + 44 * rows + (46 if subtitle else 6))


def draw_board(r, board, game, highlight=None, subtitle=None, **anchor):
    """A leaderboard card on the big screen; anchor is e.g. center=(x, y) or topright=(x, y)."""
    rect = pygame.Rect((0, 0), board_size(r, board, game, bool(subtitle)))
    setattr(rect, *next(iter(anchor.items())))
    r.shade(rect, 210, r.u(16))
    name_key, fmt = GAMES[game]
    r.text(t("board_title"), 20, YELLOW, shadow=False, midtop=(rect.centerx, rect.y + r.u(12)))
    r.text(t(name_key), 34, WHITE, midtop=(rect.centerx, rect.y + r.u(38)))
    rows = board.top(game)
    y = rect.y + r.u(92)
    line = r.u(44)
    for i, e in enumerate(rows):
        row = pygame.Rect(rect.x + r.u(14), y + i * line, rect.w - r.u(28), line - r.u(6))
        if highlight is not None and e is highlight:
            pygame.draw.rect(r.screen, YELLOW, row, border_radius=r.u(8))
            color = BLACK
        else:
            color = WHITE
        r.text(f"{i + 1}.", 24, color, shadow=False, midleft=(row.x + r.u(10), row.centery))
        r.text(e["team"], 24, color, shadow=False, midleft=(row.x + r.u(50), row.centery))
        r.text(fmt(e), 24, color, shadow=False, midright=(row.right - r.u(10), row.centery))
    if subtitle:
        r.text(subtitle, 22, YELLOW, shadow=False, midbottom=(rect.centerx, rect.bottom - r.u(12)))
