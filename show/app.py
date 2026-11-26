"""Main loop of the stage show.

A background thread reads the camera and runs the models (vision.VisionThread).
This loop draws the show at 60 fps on the big screen and the operator panel on the
laptop screen. The keys are listed in controls.py and in the panel help (H).
"""
import os
import sys
import time

import cv2

# Keep the show on the big screen when the operator clicks in the panel window.
os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")

import pygame  # noqa: E402  (the setting above must come before SDL starts)

from . import config
from .draw import GREEN, PLAYER_COLORS, RED, WHITE, YELLOW, Renderer
from .facecount import FaceCounter
from .facelock import FaceLock
from .leaderboard import Leaderboard, draw_board
from .panel import PANEL_H, PANEL_W, OperatorPanel
from .players import PlayerTracker
from .stages.attract import AttractStage
from .stages.detect import DetectStage
from .stages.evasion import EvasionStage
from .stages.hitnumber import HitNumberStage
from .stages.lesson import ClosingStage, DetectExplainStage, HideStage, OpeningStage, SignStage
from .stages.panda import PandaStage
from .stages.page_parts import STEPS
from .stages.pixels import PixelStage
from .testmode import FakeFaces, is_fake
from .texts import LANG, t, toggle_language
from .vision import Camera, Segmenter, VisionEngine, VisionThread

PANEL_FPS = 20
LIMITS = {"PLAYER_MIN_FACE": (0.02, 0.30), "FACE_VISIBLE_CONF": (0.20, 0.90), "HAAR_MIN_HITS": (2, 40),
          "MAX_PLAYERS": (1, len(PLAYER_COLORS))}
ORANGE = (255, 140, 40)


def pick_show_display():
    """The largest landscape screen that is not the laptop (display 0), or None."""
    if config.SHOW_DISPLAY is not None:
        return config.SHOW_DISPLAY
    sizes = pygame.display.get_desktop_sizes()
    others = [(w * h, i) for i, (w, h) in enumerate(sizes) if i != config.PANEL_DISPLAY and w > h]
    return max(others)[1] if others else None


class App:
    def __init__(self, camera_id, lesson=False, test=False):
        pygame.init()
        self.show_display = pick_show_display()
        self.fullscreen = self.show_display is not None or config.START_FULLSCREEN
        pygame.display.set_caption("Computing Science Day")
        self.renderer = Renderer(self._make_screen())
        self.clock = pygame.time.Clock()
        self._loading("Camera en modellen starten...")

        print("[SHOW] Loading models ...")
        self.camera = Camera(camera_id)
        self.vision = VisionThread(VisionEngine(), self.camera)
        self.vision.start()
        self.segmenter = Segmenter()  # loads only when the silhouette game is chosen

        self.panel = None
        try:
            sw = pygame.display.get_desktop_sizes()[config.PANEL_DISPLAY][0]
            x = 30 if self.show_display is not None else max(30, min(sw - PANEL_W - 20, config.FRAME_W + 40))
            self.panel = OperatorPanel(self, (x, 40))
        except Exception as e:  # the show still works with keys only
            print(f"[SHOW] Operator panel not available: {e}")

        self.qr = pygame.image.load(str(config.QR_FILE)).convert() if config.QR_FILE.exists() else None
        self.players = PlayerTracker()
        self.faces_today = FaceCounter()   # an estimate; ids and boxes only, no face data
        self.face_lock = FaceLock()        # the face the explanation pages follow (TAB: next face)
        self.stats = {"invisible": 0}
        self.board = Leaderboard()
        self.board_overlay = None     # (game, until, highlighted entry)
        self.text_input = None        # {'prompt', 'text', 'done'} while the operator types a team name
        self.operator_msg = None      # (text, until): shown in the panel only
        self._clear_armed = 0.0
        self._quit_armed = 0.0
        self.attract = AttractStage()
        pixels, detect, evasion, self.panda = PixelStage(), DetectStage(), EvasionStage(), PandaStage()
        self.show_stages = [pixels, detect, evasion, self.panda]
        # the group lesson uses the same stage objects where it can, so models load only once
        self.lesson_stages = [OpeningStage(), pixels, DetectExplainStage(), detect, HideStage(), HitNumberStage(),
                              self.panda, SignStage(), ClosingStage()]
        self.max_players = {False: config.MAX_PLAYERS, True: config.LESSON_MAX_PLAYERS}
        self.lesson = lesson
        self.stages = self.lesson_stages if lesson else self.show_stages
        config.MAX_PLAYERS = self.max_players[lesson]
        self.debug = False
        self.fake = FakeFaces()       # test mode (Y): simulated faces
        self.fake.on = test
        self.overlays = True          # header, info bars and QR code (O); game meters always show
        self.show_qr = True           # the QR code on its own (K)
        self.toast = None
        self.now = time.time()
        self.stage_start = self.now
        self.last_person = self.now
        self.frame = self.det = None
        self.index = -1
        self.stage = self.attract
        self.go(-1)

    # --- window helpers ---
    def _make_screen(self):
        """Draw at a fixed size; SCALED lets the graphics card stretch it to the window or screen.
        SCALED + FULLSCREEN is a borderless desktop fullscreen, which stays visible when the
        panel window gets the focus."""
        flags = pygame.SCALED | (pygame.FULLSCREEN if self.fullscreen else pygame.RESIZABLE)
        return pygame.display.set_mode((config.RENDER_W, config.RENDER_H), flags, display=self.show_display or 0)

    def _loading(self, msg):
        r = self.renderer
        r.screen.fill((0, 0, 0))
        r.text(msg, 40, WHITE, center=(r.size[0] // 2, r.size[1] // 2))
        pygame.display.flip()

    # --- state changes (used by keys and panel buttons) ---
    def lang_code(self):
        return LANG["current"]

    def home_name(self):
        return self.stages[0].name if self.lesson else "attract mode"

    def go(self, index):
        """Switch stage; -1 is attract mode. The group lesson has no attract mode: its start is
        the opening chapter, and after the last chapter it goes back to the opening."""
        if index >= len(self.stages):
            index = -1
        if index < 0 and self.lesson:
            index = 0
        self.index = index
        if index < 0:
            self.face_lock.reset()    # a new session: a new explanation face
        self.stage = self.attract if index < 0 else self.stages[index]
        self.stage_start = self.now
        self.stage.enter(self)
        nxt = self.stages[index + 1].name if index + 1 < len(self.stages) else self.home_name()
        print(f"[SHOW] now: {self.stage.name:<20} | RIGHT -> {nxt}")

    def set_lesson(self, lesson):
        """Switch between the pass-by show and the group lesson. Each keeps its own player limit."""
        if lesson == self.lesson:
            return
        self.max_players[self.lesson] = config.MAX_PLAYERS
        self.lesson = lesson
        config.MAX_PLAYERS = self.max_players[lesson]
        self.stages = self.lesson_stages if lesson else self.show_stages
        self.go(-1)
        self.tell_operator("Now: " + ("GROUP LESSON" if lesson else "PASS-BY SHOW"))

    def toggle_explain_detail(self):
        config.EXPLAIN_DETAIL = not config.EXPLAIN_DETAIL
        STEPS.reset()   # other texts: start the page again with no steps
        self.tell_operator("Explanations: " + ("DETAILED" if config.EXPLAIN_DETAIL else "SIMPLE"))

    def explain_page(self):
        """The explanation page on screen (None = none). A new page starts with no text steps."""
        return getattr(self.stage, "explain", None) or None

    def next_step(self):
        if self.explain_page() and STEPS.next(self.now):
            self.tell_operator(f"Explanation step {STEPS.shown} of {STEPS.total}")

    def step_back(self):
        if self.explain_page() and STEPS.back():
            self.tell_operator(f"Explanation step {STEPS.shown} of {STEPS.total}")

    def next_face(self):
        if self.det is not None and self.face_lock.next(self.det.faces, self.now):
            self.tell_operator("Explanations follow another face")
        else:
            self.tell_operator("No other face in clear view")

    def adjust(self, attr, delta):
        lo, hi = LIMITS[attr]
        setattr(config, attr, round(min(hi, max(lo, getattr(config, attr) + delta)), 2))

    def adjust_face_limit(self, up):
        """The 'face seen' limit of the active stage 3 detector: Haar hits (easy) or YOLO score (hard)."""
        if config.EVASION_MODE == "easy":
            self.adjust("HAAR_MIN_HITS", 1 if up else -1)
        else:
            self.adjust("FACE_VISIBLE_CONF", 0.05 if up else -0.05)

    def in_evasion(self):
        return isinstance(self.stage, EvasionStage)

    def toggle_evasion_mode(self):
        config.EVASION_MODE = "hard" if config.EVASION_MODE == "easy" else "easy"
        msg = t("mode_easy" if config.EVASION_MODE == "easy" else "mode_hard")
        self.tell_operator(f"Stage 3 face model: {msg}")
        if self.in_evasion():
            self.show_toast(msg)

    def show_toast(self, msg):
        self.toast = (msg, self.now + 2.5)

    def tell_operator(self, msg, seconds=4):
        self.operator_msg = (msg, self.now + seconds)

    def show_board(self, game, seconds, highlight=None):
        self.board_overlay = (game, self.now + seconds, highlight)

    def board_showing(self):
        return self.board_overlay is not None and self.now < self.board_overlay[1]

    def close_board(self):
        self.board_overlay = None

    def start_text_input(self, prompt, done):
        self.text_input = {"prompt": prompt, "text": "", "done": done}

    def quit_armed(self):
        return self.now - self._quit_armed < 3

    def request_quit(self):
        """Close the show; needs a second press within 3 seconds, so one slip does not end a demo."""
        if self.quit_armed():
            raise SystemExit
        self._quit_armed = self.now
        self.tell_operator("Click Quit (or press Q) again within 3 s to close the show.", 3)

    def clear_today(self):
        """Empty today's leaderboards; needs a second press within 3 seconds."""
        if self.now - self._clear_armed < 3:
            self.board.clear_today()
            self._clear_armed = 0.0
            self.tell_operator("Today's leaderboards are empty.")
        else:
            self._clear_armed = self.now
            self.tell_operator("Press Shift+R (or the button) again within 3 s to empty today.")

    def _text_key(self, ev):
        ti = self.text_input
        if ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.text_input = None
            ti["done"](ti["text"])
        elif ev.key == pygame.K_ESCAPE:
            self.text_input = None
        elif ev.key == pygame.K_BACKSPACE:
            ti["text"] = ti["text"][:-1]
        elif ev.unicode and ev.unicode.isprintable() and len(ti["text"]) < 24:
            ti["text"] += ev.unicode

    def handle_key(self, k, mod=0):
        if k == pygame.K_q:
            self.request_quit()
            return
        if k == pygame.K_r and mod & pygame.KMOD_SHIFT:
            self.clear_today()
            return
        if k == pygame.K_e and mod & pygame.KMOD_SHIFT:
            self.toggle_explain_detail()
            return
        if self.index < 0 and k == pygame.K_SPACE:
            self.go(0)
        elif k == pygame.K_RIGHT:
            self.go(self.index + 1)
        elif k == pygame.K_LEFT:
            self.go(max(-1, self.index - 1))
        elif k == pygame.K_ESCAPE and self.board_showing():
            self.board_overlay = None   # ESC closes a leaderboard on screen first
        elif k == pygame.K_ESCAPE:
            self.go(-1)
        elif k == pygame.K_p:
            self.go(self.stages.index(self.panda))
        elif k == pygame.K_g:
            self.set_lesson(not self.lesson)
        elif k == pygame.K_y:
            self.fake.on = not self.fake.on
            self.tell_operator("Test mode (simulated faces): " + ("ON" if self.fake.on else "OFF"))
        elif self.fake.on and k in (pygame.K_COMMA, pygame.K_PERIOD):
            if mod & pygame.KMOD_SHIFT:
                self.fake.remove_person() if k == pygame.K_COMMA else self.fake.add_person()
            else:
                self.fake.hide_one() if k == pygame.K_COMMA else self.fake.show_one()
            self.tell_operator(f"Test mode: {self.fake.total} simulated people, "
                               f"{self.fake.visible} faces in view")
        elif k == pygame.K_l:
            toggle_language()
            self.show_toast(t("language"))
        elif k == pygame.K_TAB:
            self.next_face()
        elif k in (pygame.K_PAGEDOWN, pygame.K_n) and self.explain_page():   # N: team name when no page is open
            self.next_step()
        elif k == pygame.K_PAGEUP and self.explain_page():
            self.step_back()
        elif k == pygame.K_c:
            self.vision.switch_camera()
            self.players.reset()
            self.face_lock.reset()
            self.show_toast("...")
        elif k == pygame.K_f:
            self.fullscreen = not self.fullscreen
            pygame.display.toggle_fullscreen()
        elif k == pygame.K_d:
            self.debug = not self.debug
        elif k == pygame.K_k:
            self.show_qr = not self.show_qr
            self.tell_operator("QR code: " + ("ON" if self.show_qr else "OFF"))
        elif k == pygame.K_o:
            self.overlays = not self.overlays
            self.tell_operator("Overlays (header, info bars, QR code): " + ("ON" if self.overlays else "OFF"))
        elif k == pygame.K_h:
            if self.panel:
                self.panel.help = not self.panel.help
        elif k == pygame.K_m:
            self.toggle_evasion_mode()
        elif k == pygame.K_v:
            config.SHOW_SCORES = not config.SHOW_SCORES
            self.tell_operator("Face scores on screen: " + ("ON" if config.SHOW_SCORES
                                                            else "OFF (stage 3: only during a round)"))
        elif self.debug and k in (pygame.K_LEFTBRACKET, pygame.K_RIGHTBRACKET):
            self.adjust("PLAYER_MIN_FACE", 0.01 if k == pygame.K_RIGHTBRACKET else -0.01)
        elif self.debug and k in (pygame.K_MINUS, pygame.K_EQUALS):
            self.adjust_face_limit(k == pygame.K_EQUALS)
        else:
            self.stage.on_key(k, self)

    def handle_event(self, ev):
        on_panel = self.panel is not None and ev.type in (pygame.MOUSEBUTTONDOWN, pygame.WINDOWCLOSE) \
            and self.panel.owns(ev)
        if ev.type == pygame.QUIT:
            raise SystemExit
        if ev.type == pygame.WINDOWCLOSE and not on_panel:
            raise SystemExit
        if ev.type == pygame.WINDOWCLOSE and on_panel:
            self.panel.window.minimize()  # the panel cannot be closed, only hidden
        elif ev.type == pygame.KEYDOWN and self.text_input is not None:
            self._text_key(ev)
        elif ev.type == pygame.KEYDOWN:
            self.handle_key(ev.key, ev.mod)
        elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 and on_panel:
            self.panel.click(ev.pos)

    # --- main loop ---
    def run(self):
        t_prev = time.time()
        self._page = None
        last_seq, last_det_time, last_panel, last_cam = 0, t_prev, 0.0, -1
        video = None
        while True:
            self.now = time.time()
            dt, t_prev = self.now - t_prev, self.now
            for ev in pygame.event.get():
                self.handle_event(ev)
            page = (self.stage, self.explain_page())
            if page != self._page:     # another explanation page: its text waits for the presenter
                self._page = page
                STEPS.reset()
            # the Haar cascade costs time, so it only runs where it plays
            self.vision.engine.haar_on = self.in_evasion() and config.EVASION_MODE == "easy"

            seq, latest = self.vision.latest()
            if latest is None:
                self._loading("Camera en modellen starten...")
                self.clock.tick(30)
                continue
            if seq != last_seq:  # a new detection result: update the game state once
                last_seq = seq
                self.frame, self.det = latest
                self.det = self.fake.apply(self.det)
                det_dt, last_det_time = self.now - last_det_time, self.now
                self._on_detections(self.det, det_dt)
                if self.toast and self.toast[0] == "...":
                    self.show_toast(t("camera", n=self.camera.index))
            self.stage.animate(self, self.now, dt)

            # the video follows the camera, not the AI: smooth image, boxes from the newest AI result
            cam_n, raw = self.camera.peek()
            if cam_n != last_cam and raw is not None:
                video, last_cam = cv2.flip(raw, 1), cam_n
            r = self.renderer
            r.video(video if video is not None else self.frame)
            show_header = self.overlays and self.stage.title_key is not None
            r.top = r.u(config.HEADER_H) if show_header else 0
            self.stage.draw(r, self, self.det, self.now)
            if show_header:
                self.stage.draw_header(r, self)
            self.draw_overlays(r, self.det)
            pygame.display.flip()

            if self.panel and self.now - last_panel >= 1 / PANEL_FPS:
                last_panel = self.now
                self.panel.draw()
            self.clock.tick(config.SHOW_FPS)

    def _on_detections(self, det, dt):
        self.faces_today.update([f for f in det.faces if f.conf >= config.FACE_VISIBLE_CONF and not is_fake(f)],
                                self.now)
        if det.faces or det.persons:
            self.last_person = self.now
        if self.index >= 0 and not self.lesson and self.now - self.last_person > config.IDLE_RETURN:
            self.go(-1)
        self.players.update(det, self.now)
        self.face_lock.update(det.faces, self.now)
        self.stage.update(self, det, self.now, dt)

    def draw_overlays(self, r, det):
        sw, sh = r.size
        if self.overlays and self.show_qr and self.qr is not None and getattr(self.stage, "game", None) is None                 and not getattr(self.stage, "big_qr", False) and not getattr(self.stage, "explain", None):
            q = r.u(90)   # bottom-right corner, out of the way of the explanations; no QR over a game
            r.screen.blit(pygame.transform.smoothscale(self.qr, (q, q)), (sw - q - r.u(6), sh - q - r.u(6)))
        if self.board_overlay and self.now < self.board_overlay[1]:
            game, _, hl = self.board_overlay
            draw_board(r, self.board, game, highlight=hl, center=(sw // 2, sh // 2))
        if self.text_input is not None:
            card = pygame.Rect(sw // 2 - r.u(400), sh // 2 - r.u(90), r.u(800), r.u(180))
            r.shade(card, 225, r.u(18))
            r.text(self.text_input["prompt"], 28, YELLOW, midtop=(card.centerx, card.y + r.u(20)))
            cursor = "_" if int(self.now * 2) % 2 else " "
            r.text(self.text_input["text"] + cursor, 56, WHITE, midtop=(card.centerx, card.y + r.u(70)))
        if self.toast and self.now < self.toast[1] and self.toast[0] != "...":
            r.text(self.toast[0], 36, WHITE, center=(sw // 2, sh // 2 - r.u(180)))
        if self.debug:
            lines = [
                f"screen {self.clock.get_fps():5.1f}  camera {self.camera.fps:5.1f}  AI {self.vision.fps:5.1f} fps   "
                f"face {det.face_ms:5.1f} ms   objects {det.object_ms:5.1f} ms   haar {det.haar_ms:5.1f} ms",
                f"{'lesson' if self.lesson else 'show'}: {self.stage.name}   camera {self.camera.index}   stage 3 mode {config.EVASION_MODE}",
                f"faces {len(det.faces)}  persons {len(det.persons)}  objects {len(det.objects)}"
                f"  players {len(self.players.players)}",
                f"[ ] player_min_face {config.PLAYER_MIN_FACE:.2f}   - = face_visible {config.FACE_VISIBLE_CONF:.2f}"
                f"  haar_min_hits {config.HAAR_MIN_HITS}",
            ]
            for p in self.players.players.values():
                lines.append(f"  P{p.body_id}: body={'yes' if p.body else 'NO '} face={p.face_conf:.2f} "
                             f"haar={p.haar_hits:2d} invisible={p.invisible!s:5} streak={p.streak:4.1f}")
            pad = pygame.Rect(r.u(10), sh - r.u(80) - r.u(22) * len(lines), r.u(700), r.u(22) * len(lines) + r.u(10))
            r.shade(pad, 200)
            for i, s in enumerate(lines):
                r.text(s, 15, GREEN, shadow=False, bold=False, mono=True,
                       topleft=(pad.x + r.u(8), pad.y + r.u(5) + i * r.u(22)))
            for o in det.persons:
                pygame.draw.rect(r.screen, RED, r.rect(o.box), 1)
            for box, why in self.players.why_not:   # why this person is not a player
                r.text(why, 15, ORANGE, bold=False, topleft=r.rect(box).move(r.u(4), r.u(4)).topleft)
        if self.fake.on:   # always on screen, also with the overlays off: nobody may take it for real
            self.fake.draw(r, det)


def main():
    """python -m show [camera number] [--lesson] [--test]"""
    args = sys.argv[1:]
    rest = [a for a in args if not a.startswith("--")]
    camera_id = int(rest[0]) if rest else config.CAMERA_ID
    app = App(camera_id, lesson="--lesson" in args, test="--test" in args)
    try:
        app.run()
    except SystemExit:
        pass
    finally:
        app.vision.stop()
        pygame.quit()
