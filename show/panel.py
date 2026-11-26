"""Operator panel: a second window on the laptop screen. It is a fallback for the keys:
every button sends the same key the operator could press, so both always agree.

Pages: Control (navigation and stage actions), Scores (leaderboards) and Help (all keys).
"""
import pygame

from . import config
from .controls import SHORTCUTS
from .leaderboard import GAMES
from .stages.detect import DetectStage
from .stages.hitnumber import HitNumberStage
from .texts import object_name

PANEL_W, PANEL_H = 460, 760
BG = (24, 26, 32)
CARD = (38, 41, 50)
BTN = (58, 63, 78)
BTN_MAIN = (40, 120, 220)
BTN_ON = (90, 70, 30)
BTN_DANGER = (120, 45, 45)
WHITE = (235, 235, 240)
DIM = (150, 155, 170)
YELLOW = (255, 210, 60)
GREEN = (60, 220, 120)


class Button:
    def __init__(self, rect, key=None, action=None):
        self.rect, self.key, self.action = pygame.Rect(rect), key, action


class OperatorPanel:
    def __init__(self, app, position):
        from pygame._sdl2.video import Renderer, Texture, Window
        self.app = app
        self.window = Window("Operator panel", size=(PANEL_W, PANEL_H), position=position)
        self.renderer = Renderer(self.window)
        self._Texture = Texture
        self.texture = None
        self.surface = pygame.Surface((PANEL_W, PANEL_H))
        self.page = "control"   # control | scores
        self.help = False
        self.buttons = []
        self.f_title = pygame.font.SysFont("arial", 22, bold=True)
        self.f = pygame.font.SysFont("arial", 17, bold=True)
        self.f_small = pygame.font.SysFont("arial", 15)
        self.f_mono = pygame.font.SysFont("consolas", 15, bold=True)
        self._help_fonts = {}

    # --- input ---
    def owns(self, event):
        w = getattr(event, "window", None)
        if w is not None and hasattr(w, "id"):
            return w.id == self.window.id
        return not pygame.mouse.get_focused()  # the show window does not have the mouse

    def click(self, pos):
        for b in self.buttons:
            if b.rect.collidepoint(pos):
                if b.action:
                    b.action()
                elif b.key is not None:
                    self.app.handle_key(b.key)
                return

    # --- drawing helpers ---
    def _text(self, s, font, color, pos):
        img = font.render(s, True, color)
        self.surface.blit(img, pos)
        return img.get_rect(topleft=pos)

    def _button(self, rect, label, key=None, action=None, color=BTN, hint=None, font=None):
        b = Button(rect, key, action)
        self.buttons.append(b)
        pygame.draw.rect(self.surface, color, b.rect, border_radius=8)
        img = (font or self.f).render(label, True, WHITE)
        self.surface.blit(img, img.get_rect(center=b.rect.center))
        if hint:
            h = self.f_small.render(hint, True, DIM)
            self.surface.blit(h, h.get_rect(topright=(b.rect.right - 6, b.rect.y + 2)))

    def _set_page(self, page):
        self.page, self.help = page, False

    def draw(self):
        self.buttons = []
        self.surface.fill(BG)
        x, w = 16, PANEL_W - 32
        # tabs, and Quit on every page
        tab_w = (w - 30) // 4
        for i, (label, page) in enumerate((("Control", "control"), ("Scores", "scores"))):
            on = not self.help and self.page == page
            self._button((x + i * (tab_w + 10), 10, tab_w, 34), label, action=lambda p=page: self._set_page(p),
                         color=BTN_MAIN if on else BTN)
        self._button((x + 2 * (tab_w + 10), 10, tab_w, 34), "Help", key=pygame.K_h,
                     color=BTN_MAIN if self.help else BTN, hint="H")
        armed = self.app.quit_armed()
        self._button((x + 3 * (tab_w + 10), 10, tab_w, 34), "Click again" if armed else "Quit",
                     action=self.app.request_quit, color=(200, 60, 60) if armed else BTN_DANGER,
                     font=self.f_small if armed else self.f)

        if self.help:
            self._draw_help(x, w)
        elif self.page == "scores":
            self._draw_scores(x, w)
        else:
            self._draw_control(x, w)

        msg = self.app.operator_msg
        if msg and self.app.now < msg[1]:
            bar = pygame.Rect(0, PANEL_H - 40, PANEL_W, 40)
            pygame.draw.rect(self.surface, (90, 70, 20), bar)
            self._text(msg[0], self.f_small, WHITE, (x, bar.y + 11))

        if self.texture is None:
            self.texture = self._Texture.from_surface(self.renderer, self.surface)
        else:
            self.texture.update(self.surface)
        self.renderer.clear()
        self.texture.draw()
        self.renderer.present()

    # --- pages ---
    def _draw_control(self, x, w):
        app, s = self.app, self.surface
        card = pygame.Rect(x, 56, w, 132)
        pygame.draw.rect(s, CARD, card, border_radius=10)
        n = len(app.stages)
        now = f"{app.index + 1}/{n}  {app.stage.name}" if app.index >= 0 else "attract mode"
        if app.lesson:
            now = "LESSON " + now
        nxt = app.stages[app.index + 1].name if app.index + 1 < n else app.home_name()
        lines = [
            ("Now", now, YELLOW),
            ("Next (RIGHT)", nxt, WHITE),
            ("Players", f"{len(app.players.players)}/{config.MAX_PLAYERS}   camera {app.camera.index}   language "
                        f"{app.lang_code().upper()}", WHITE),
            ("Speed", f"screen {app.clock.get_fps():3.0f}   camera {app.camera.fps:4.1f}   "
                      f"AI {app.vision.fps:4.1f} fps", WHITE),
        ]
        for i, (k, v, c) in enumerate(lines):
            self._text(k, self.f_small, DIM, (card.x + 12, card.y + 10 + i * 30))
            self._text(v, self.f, c, (card.x + 130, card.y + 9 + i * 30))

        y = card.bottom + 12
        if app.board_showing():   # a leaderboard on screen: one click (or ESC) closes it
            self._button((x, y, w, 50), "Close leaderboard on screen", action=app.close_board,
                         color=(40, 140, 80), hint="ESC")
            y += 58
        action = app.stage.action_label(app)
        if action:
            self._button((x, y, w, 50), action, key=pygame.K_SPACE, color=BTN_MAIN, hint="SPACE")
        else:
            pygame.draw.rect(s, CARD, (x, y, w, 50), border_radius=8)
            img = self.f.render("no action right now", True, DIM)
            s.blit(img, img.get_rect(center=(x + w // 2, y + 25)))
        y += 58
        if app.stage.round_running(app):
            self._button((x, y, w, 40), f"+{config.EXTEND_TIME:.0f} s more time", key=pygame.K_x,
                         color=(40, 140, 80), hint="X")
            y += 48

        stage2 = isinstance(app.stage, DetectStage)
        if stage2:  # stage 2: game choice
            y = self._draw_stage2_games(x, w, y)
        if isinstance(app.stage, HitNumberStage):
            y = self._draw_hit_number(app.stage, x, w, y)
        if app.in_evasion():  # stage 3: which face model plays
            easy = config.EVASION_MODE == "easy"
            self._button((x, y, w, 36), "Mode: EASY (Haar cascade)" if easy else "Mode: HARD (YOLO)",
                         key=pygame.K_m, color=BTN_ON if easy else BTN, hint="M")
            y += 44
        holder = app.stage.game if stage2 else app.stage
        if holder is not None and hasattr(holder, "instructions"):
            self._button((x, y, w, 36), "Hide instructions" if holder.instructions else "Show instructions",
                         key=pygame.K_i, color=BTN_ON if holder.instructions else BTN, hint="I")
            y += 44
        if hasattr(app.stage, "explain") and getattr(app.stage, "game", None) is None:  # stage 2 (free), 3, 4
            exp = app.stage.explain
            label = {"filter": "Explanation: a filter  (E: close)",
                     "old": "Explanation: old AI  (E: new AI)", "new": "Explanation: new AI  (E: grid)",
                     "grid": "Explanation: clues to boxes  (E: close)",
                     "fail": "Explanation: why the AI misses  (E: close)",
                     None: "Show explanation"}.get(exp) if not isinstance(exp, bool) \
                else ("Hide explanation" if exp else "Show explanation")
            self._button((x, y, w, 36), label,
                         key=pygame.K_e, color=BTN_ON if app.stage.explain else BTN, hint="E")
            y += 44
            half = (w - 10) // 2
            self._button((x, y, half, 36), "Level: DETAILED" if config.EXPLAIN_DETAIL else "Level: SIMPLE",
                         action=app.toggle_explain_detail, color=BTN_ON if config.EXPLAIN_DETAIL else BTN,
                         hint="Shift+E")
            self._button((x + half + 10, y, half, 36), "Next face", action=app.next_face, hint="TAB")
            y += 44
            if app.stage.explain:
                from .stages.page_parts import STEPS
                self._button((x, y, half, 36), "< Step back", action=app.step_back, hint="PgUp")
                self._button((x + half + 10, y, half, 36), f"Next step  {STEPS.shown}/{STEPS.total}",
                             action=app.next_step, color=BTN_ON if STEPS.shown < STEPS.total else BTN,
                             hint="PgDn / N")
                y += 44

        for row in ((("< Previous", pygame.K_LEFT, "LEFT"), ("Next >", pygame.K_RIGHT, "RIGHT")),
                    (("Opening" if app.lesson else "Attract mode", pygame.K_ESCAPE, "ESC"),
                     ("Panda backup", pygame.K_p, "P"),
                     ("To show" if app.lesson else "To lesson", pygame.K_g, "G")),
                    (("Language NL/EN", pygame.K_l, "L"), ("Next camera", pygame.K_c, "C")),
                    (("Overlays", pygame.K_o, "O"), ("QR", pygame.K_k, "K"), ("Full", pygame.K_f, "F"),
                     ("Debug", pygame.K_d, "D"), ("Scores", pygame.K_v, "V"))):
            bw = (w - 10 * (len(row) - 1)) // len(row)
            for i, (label, key, hint) in enumerate(row):
                on = ((key == pygame.K_d and app.debug) or (key == pygame.K_v and config.SHOW_SCORES)
                      or (key == pygame.K_o and app.overlays) or (key == pygame.K_k and app.show_qr))
                self._button((x + i * (bw + 10), y, bw, 40), label, key=key, hint=hint, color=BTN_ON if on else BTN)
            y += 48

        if stage2:
            return  # stage 2 needs the space for the game controls; the room settings are for players
        y += 8
        self._text("ROOM SETTINGS", self.f_small, DIM, (x, y))
        y += 22
        if config.EVASION_MODE == "easy":
            limit = ("Haar hits needed", f"{config.HAAR_MIN_HITS}")
        else:
            limit = ("Face visible limit", f"{config.FACE_VISIBLE_CONF:.2f}")
        rows = (("Player face size", f"{config.PLAYER_MIN_FACE:.2f}",
                 lambda: app.adjust("PLAYER_MIN_FACE", -0.01), lambda: app.adjust("PLAYER_MIN_FACE", 0.01)),
                (*limit, lambda: app.adjust_face_limit(False), lambda: app.adjust_face_limit(True)),
                ("Max players", f"{config.MAX_PLAYERS}",
                 lambda: app.adjust("MAX_PLAYERS", -1), lambda: app.adjust("MAX_PLAYERS", 1)))
        for label, value, down, up in rows:
            self._text(label, self.f, WHITE, (x, y + 8))
            self._text(value, self.f_mono, YELLOW, (x + 200, y + 9))
            self._button((x + w - 110, y, 50, 34), "-", action=down)
            self._button((x + w - 50, y, 50, 34), "+", action=up)
            y += 42
        if app.debug:   # why people in view are not players (also on the show screen)
            self._text("NOT A PLAYER, BECAUSE ...", self.f_small, DIM, (x, y + 4))
            y += 24
            reasons = [why for _, why in app.players.why_not] or ["(everyone in view is a player)"]
            for why in reasons[:4]:
                self._text("- " + why, self.f_small, WHITE, (x, y))
                y += 19

    def _draw_hit_number(self, stage, x, w, y):
        """Hit the number: the target (UP / DOWN) and measuring the maximum again (BACKSPACE)."""
        self._text("HIT THE NUMBER", self.f_small, DIM, (x, y))
        y += 20
        third = (w - 20) // 3
        self._button((x, y, third, 34), "-1", key=pygame.K_DOWN, hint="DOWN")
        top = "?" if stage.maximum is None else stage.maximum
        img = self.f.render(f"{stage.target}  (max {top})", True, YELLOW)
        self.surface.blit(img, img.get_rect(center=(x + third + 10 + third // 2, y + 17)))
        self._button((x + 2 * (third + 10), y, third, 34), "+1", key=pygame.K_UP, hint="UP")
        y += 42
        self._button((x, y, w, 36), "Measure the maximum again", key=pygame.K_BACKSPACE, hint="BACKSPACE")
        return y + 44

    def _draw_stage2_games(self, x, w, y):
        stage = self.app.stage
        game = stage.game
        self._text("STAGE 2 GAME", self.f_small, DIM, (x, y))
        y += 20
        bw = (w - 40) // 5
        options = [("Free", pygame.K_0, None), ("1 Fool", pygame.K_1, "fool"),
                   ("2 Guess", pygame.K_2, "guess"), ("3 Hunt", pygame.K_3, "hunt"), ("4 Confuse", pygame.K_4, "confuse")]
        for i, (label, key, gid) in enumerate(options):
            active = (game is None and gid is None) or (game is not None and game.game_id == gid)
            color = BTN_MAIN if active else (CARD if key is None else BTN)
            self._button((x + i * (bw + 10), y, bw, 34), label, key=key, color=color, font=self.f_small)
        y += 42
        if game is not None:
            if game.game_id == "fool":  # the target, in a fixed order
                third = (w - 20) // 3
                self._button((x, y, third, 34), "< Target", action=lambda: game.step_target(-1), hint="Sh+T")
                img = self.f.render(object_name(game.target), True, YELLOW)
                s = self.surface
                s.blit(img, img.get_rect(center=(x + third + 10 + third // 2, y + 17)))
                self._button((x + 2 * (third + 10), y, third, 34), "Target >", action=lambda: game.step_target(1),
                             hint="T")
                y += 42
            if game.game_id == "confuse" and game.phase in ("wait", "result"):
                y = self._draw_confuse_choices(game, x, w, y)
            if game.can_name():
                self._button((x, y, w, 40), "Enter team name", key=pygame.K_n, color=(40, 140, 80), hint="N")
                y += 48
        return y + 6

    def _draw_confuse_choices(self, game, x, w, y):
        """The numbered candidates as buttons: click one to choose only that one, click it again
        for the automatic choice (T: next number, after the last one automatic)."""
        choices = game.choices()
        pick = "your choice, click again for auto" if game.pick_id is not None else "auto: all held"
        self._text(f"OBJECTS THAT COUNT ({pick}; * = held)", self.f_small, DIM, (x, y))
        y += 20
        if not choices:
            self._text("no objects in view", self.f_small, DIM, (x, y + 4))
            return y + 30
        cols = 3
        bw = (w - 10 * (cols - 1)) // cols
        counted = game.counted_ids()
        for i, (n, o, held) in enumerate(choices):
            bx, by = x + (i % cols) * (bw + 10), y + (i // cols) * 38
            label = f"{n} {object_name(o.name)}{' *' if held else ''}"
            self._button((bx, by, bw, 32), label, action=lambda oid=o.id: game.pick(oid),
                         color=BTN_MAIN if o.id in counted else BTN, font=self.f_small)
        return y + ((len(choices) + cols - 1) // cols) * 38 + 4

    def _draw_scores(self, x, w):
        app, s = self.app, self.surface
        y = 58
        for game, (name_key, fmt) in GAMES.items():
            self._text(_game_label(game), self.f_title, WHITE, (x, y))
            self._button((x + w - 150, y - 2, 150, 30), "Show on screen", font=self.f_small,
                         action=lambda g=game: app.show_board(g, 10))
            y += 36
            rows = app.board.top(game)
            if not rows:
                self._text("no scores today", self.f_small, DIM, (x, y))
                y += 30
            for i, e in enumerate(rows):
                row = pygame.Rect(x, y, w, 32)
                pygame.draw.rect(s, CARD, row, border_radius=6)
                self._text(f"{i + 1}. {e['team']}", self.f, WHITE, (row.x + 10, row.y + 6))
                self._text(fmt(e), self.f_small, YELLOW, (row.x + 220, row.y + 8))
                self._button((row.right - 34, row.y + 3, 30, 26), "x", color=BTN_DANGER,
                             action=lambda en=e: app.board.remove(en))
                y += 38
            y += 14
        if app.board_showing():
            self._button((x, y, w, 34), "Close leaderboard on screen", action=app.close_board,
                         color=(40, 140, 80), hint="ESC")
            y += 42
        self._text("x = delete an entry (for example a rude team name)", self.f_small, DIM, (x, y))
        y += 34
        self._button((x, y, w, 40), "Empty today's leaderboards", key=None, color=BTN_DANGER,
                     action=app.clear_today, hint="Shift+R")

    def _draw_help(self, x, w):
        """All keys; the text gets smaller until the list fits the panel."""
        for size in (15, 14, 13, 12, 11):
            font = self._help_fonts.setdefault(size, pygame.font.SysFont("arial", size))
            rows = [_wrap(text, font, w - 110) for _, _, text in SHORTCUTS]
            line = size + 3
            if 58 + sum(len(r) * line + 4 for r in rows) + 30 <= PANEL_H - 40:
                break
        y = 58
        for (_, key_name, _), lines in zip(SHORTCUTS, rows):
            self._text(key_name, self.f_mono, YELLOW, (x, y))
            for text_line in lines:
                self._text(text_line, font, WHITE, (x + 110, y))
                y += line
            y += 4
        self._text("All keys work in both windows.", self.f_small, GREEN, (x, y + 4))


def _game_label(game):
    return {"fool": "1 Fool the AI", "hunt": "3 Scavenger hunt", "confuse": "4 Confusion meter"}.get(game, game)


def _wrap(text, font, width):
    words, lines, cur = text.split(), [], ""
    for word in words:
        test = f"{cur} {word}".strip()
        if font.size(test)[0] <= width:
            cur = test
        else:
            lines.append(cur)
            cur = word
    return lines + [cur] if cur else lines
