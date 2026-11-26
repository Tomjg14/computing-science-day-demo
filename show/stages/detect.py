"""Stage 2 - How sure is the computer? Faces and objects, colored by their confidence score.

An object is only shown after the tracker has followed it for a few frames in a row,
so one-frame mistakes do not reach the screen. To keep the screen calm, at most
config.STAGE2_MAX_OBJECTS objects show: objects held by a player first, then the largest.
Static objects (furniture and the like) never show.

Key E in free detection: three explanation pages, live with real numbers. The old AI (Haar
cascade) and the modern AI (layers of filters) on a face (explain_pages.py), then how the
detector turns these clues into boxes (detect_explain.py). E after the last page closes them.
"""
import pygame

from .. import config
from ..games.confuse import ConfuseGame
from ..games.fool import FoolGame
from ..games.guess import GuessGame
from ..games.hunt import HuntGame
from ..draw import GREEN, WHITE, YELLOW
from ..players import is_big, rank_objects
from ..texts import object_name, t
from .base import Stage
from .detect_explain import DetectExplainer
from .explain_pages import ModelExplainer

ORANGE = (255, 140, 40)
MAX_FOUND = 8
GAME_KEYS = {pygame.K_1: FoolGame, pygame.K_2: GuessGame, pygame.K_3: HuntGame, pygame.K_4: ConfuseGame}
PAGES = [None, "old", "new", "grid"]   # E goes to the next one; after "grid" back to None


def conf_color(conf):
    if conf >= config.CONF_SURE:
        return GREEN
    return YELLOW if conf >= config.CONF_UNSURE else ORANGE


class DetectStage(Stage):
    name = "detection"
    title_key = "s2_title"
    sub_key = "s2_sub"

    def enter(self, app):
        self.found = {}    # English class name -> best confidence of a stable detection
        self.streak = {}   # track id -> frames in a row this object was seen
        self.stable = []
        self.game = None   # None = free detection, else one of the detection games
        if DetectStage.explainer is None:   # made once, also for the lesson chapter: they load models
            DetectStage.explainer = DetectExplainer()
            DetectStage.models = ModelExplainer()
        self.explainer.preload()
        self.models.preload()
        self.page = None
        self._open(None, app.now)

    explainer = None       # page "grid": how YOLO turns clues into boxes
    models = None          # pages "old" and "new": how the old and the modern AI look at a face

    @property
    def explain(self):
        """The explanation page on screen: None, "old", "new" or "grid" (for the operator panel);
        only in free detection."""
        return self.page if self.game is None else None

    def _open(self, page, now):
        self.page = page
        self.models.open(page if page in ("old", "new") else None, now)
        if (page == "grid") != self.explainer.on:
            self.explainer.toggle(now)

    def action_label(self, app):
        return self.game.action_label() if self.game else None

    def round_running(self, app):
        return getattr(self.game, "round_running", lambda: False)()

    def on_key(self, key, app):
        if key in GAME_KEYS:
            self.game = GAME_KEYS[key](app)
            self._open(None, app.now)
            return True
        if key == pygame.K_0:
            self.game = None
            return True
        if key == pygame.K_e and self.game is None:
            self._open(PAGES[(PAGES.index(self.page) + 1) % len(PAGES)], app.now)
            return True
        return self.game.on_key(key, app) if self.game else False

    def update(self, app, det, now, dt):
        seen = {}
        for o in det.objects:
            if o.id is not None:
                seen[o.id] = self.streak.get(o.id, 0) + 1
        self.streak = seen
        stable = [o for o in det.objects if o.id is not None and seen[o.id] >= config.OBJECT_MIN_FRAMES]
        self.stable = rank_objects(stable, app.players.players.values())[:config.STAGE2_MAX_OBJECTS]
        for o in self.stable:
            self.found[o.name] = max(o.conf, self.found.get(o.name, 0.0))
        if self.game:
            self.game.update(app, det, now, dt)
        elif self.page in ("old", "new"):
            self.models.update(app, det, now)
        elif self.page == "grid":
            self.explainer.update(app)

    def draw(self, r, app, det, now):
        if self.game:
            self.game.draw(r, app, det, now)
            return
        if self.explain in ("old", "new"):
            self.models.draw(r, app, now)
            return
        if self.explain == "grid":
            self.explainer.draw(r, app, now)
            return
        for f in det.faces:
            if f.conf >= config.FACE_VISIBLE_CONF and is_big(f, det.frame_w):
                if config.SHOW_SCORES:  # the color also tells the score, so both go together
                    r.labeled_box(f.box, f"{t('face')} {f.conf:.0%}", conf_color(f.conf))
                else:
                    r.labeled_box(f.box, t("face"), GREEN)
        for o in self.stable:
            r.labeled_box(o.box, f"{object_name(o.name)} {o.conf:.0%}", conf_color(o.conf), size=24)

        if not app.overlays:
            return
        sw, sh = r.size
        if self.found:
            items = sorted(self.found.items(), key=lambda kv: -kv[1])[:MAX_FOUND]
            line = r.u(36)
            panel = pygame.Rect(sw - r.u(330), r.top + r.u(20), r.u(310), r.u(60) + line * len(items))
            r.shade(panel, 170, r.u(14))
            r.text(t("s2_found"), 26, YELLOW, topleft=(panel.x + r.u(16), panel.y + r.u(12)))
            for i, (name, conf) in enumerate(items):
                y = panel.y + r.u(52) + i * line
                r.text(object_name(name), 22, WHITE, topleft=(panel.x + r.u(16), y))
                r.text(f"{conf:.0%}", 22, conf_color(conf), topright=(panel.right - r.u(16), y))

        # legend: what the colors mean
        r.banner(sh - r.u(100), r.u(100))
        x = r.u(40)
        for label, color in zip(t("s2_legend"), (GREEN, YELLOW, ORANGE)):
            pygame.draw.rect(r.screen, color, (x, sh - r.u(88), r.u(26), r.u(26)), border_radius=r.u(5))
            x = r.text(label, 22, WHITE, shadow=False, topleft=(x + r.u(36), sh - r.u(88))).right + r.u(40)
        r.text(t("s2_hint"), 24, WHITE, center=(sw // 2, sh - r.u(30)))
