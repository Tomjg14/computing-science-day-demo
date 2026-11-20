"""Base class for the attract mode and the session stages."""
import pygame

from .. import config
from ..draw import GREEN, GREY, YELLOW
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

    def draw(self, r, app, det, now):
        pass

    def draw_header(self, r, app):
        if not self.title_key:
            return
        sw, _ = r.size
        r.banner(0, r.u(108))
        r.text(t("step", n=app.index + 1, total=len(app.stages)), 18, YELLOW, midtop=(sw // 2, r.u(6)))
        r.text(t(self.title_key), 46, midtop=(sw // 2, r.u(28)))
        r.text(t(self.sub_key), 20, shadow=False, bold=False, midtop=(sw // 2, r.u(80)))


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
