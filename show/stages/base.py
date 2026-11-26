"""Base class for the attract mode and the session stages."""
import pygame

from .. import config
from ..draw import GREEN, GREY, WHITE, YELLOW
from ..players import is_big
from ..texts import t


class Stage:
    name = "stage"           # shown in the operator console
    title_key = None         # text keys for the header; None = no header
    sub_key = None

    def enter(self, app):
        """Called each time the show switches to this stage."""

    def on_key(self, key, app):
        """Handle a stage-specific key; return True when used."""
        return False

    def update(self, app, det, now, dt):
        """Called once per new detection result (about 10-15 times per second)."""

    def animate(self, app, now, dt):
        """Called once per screen frame (60 per second), for smooth effects."""

    def action_label(self, app):
        """What SPACE does in this stage, for the operator panel; None = nothing."""
        return None

    def round_running(self, app):
        """True while a timed round runs, so X can add time (for the operator panel)."""
        return False

    def draw(self, r, app, det, now):
        pass

    def draw_header(self, r, app):
        """One thin line at the top: step and title. The subtitle shows below it for the first
        seconds of the stage, then fades out, so the header covers little of the camera image."""
        if not self.title_key:
            return
        sw, _ = r.size
        h = r.u(config.HEADER_H)
        r.banner(0, h, 110)
        x = r.text(t("step", n=app.index + 1, total=len(app.stages)) + "  ·  ", 20, YELLOW,
                   midleft=(r.u(20), h // 2)).right
        r.text(t(self.title_key), 26, WHITE, midleft=(x, h // 2))
        el = app.now - app.stage_start
        fade = 1.0 if el < config.SUBTITLE_TIME - 1 else max(0.0, config.SUBTITLE_TIME - el)
        if fade > 0 and self.sub_key and getattr(self, "game", None) is None and not getattr(self, "explain", None):
            # a game has its own task line there, and an explanation its own title
            r.text(t(self.sub_key), 20, WHITE, alpha=int(255 * fade), topleft=(r.u(20), h + r.u(6)))


def draw_faces(r, det, skip=(), scores=None):
    """Grey thin box for passers-by, green labeled box for faces close to the screen.
    scores: show the confidence in the label; None = follow the V toggle."""
    if scores is None:
        scores = config.SHOW_SCORES
    for f in det.faces:
        if id(f) in skip or f.conf < config.FACE_VISIBLE_CONF:
            continue
        if is_big(f, det.frame_w):
            r.labeled_box(f.box, f"{t('face')} {f.conf:.0%}" if scores else t("face"), GREEN)
        else:
            pygame.draw.rect(r.screen, GREY, r.rect(f.box), r.u(2), border_radius=r.u(6))
