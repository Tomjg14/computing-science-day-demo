"""Attract mode: passers-by see themselves with face boxes and an invitation to come closer."""
from .. import config
from ..leaderboard import draw_board
from ..draw import WHITE, YELLOW
from ..texts import t
from .base import Stage, draw_faces


class AttractStage(Stage):
    name = "attract mode"

    def action_label(self, app):
        return "Start session"

    def draw(self, r, app, det, now):
        draw_faces(r, det)
        sw, sh = r.size
        r.banner(sh - r.u(150), r.u(150))
        invites = t("invites")
        invite = invites[int(now / config.INVITE_SWITCH) % len(invites)]
        r.text(invite, 64, WHITE, center=(sw // 2, sh - r.u(95)))
        r.text(t("faces_today", n=len(app.stats["faces_seen"])), 26, YELLOW, center=(sw // 2, sh - r.u(32)))

        # now and then: a leaderboard with a challenge, on the right side
        games = app.board.games_with_entries()
        if games and now % config.ATTRACT_BOARD_EVERY < config.ATTRACT_BOARD_TIME:
            game = games[int(now // config.ATTRACT_BOARD_EVERY) % len(games)]
            best = app.board.best(game)
            draw_board(r, app.board, game, subtitle=t("beat", team=best["team"]),
                       topright=(sw - r.u(30), r.u(30)))
