"""
PROTOTYPE - throwaway code, not part of the real demo.

Question it answers: does "body seen, face not seen = invisible" work reliably
with real people and accessories, and is the screen fun and readable?

Run from the repo root:
    conda activate demo_day
    python prototype_evasion/evasion_prototype.py

Keys:
    SPACE   start session / start a new round
    ESC     back to attract mode
    D       debug overlay on/off
    F       fullscreen on/off
    [ ]     player face size limit down/up
    - =     face visible threshold down/up
    C       switch to the next camera (e.g. an external USB webcam)
    Q       quit
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # same workaround as main.py

import math
import os
import random
import sys
import time

import cv2
import numpy as np
import pygame
import requests
from ultralytics import YOLO

# --- settings (tune live with keys, see docstring) ---
CAMERA_ID = 0               # start camera; override with: python evasion_prototype.py 1
MAX_CAMERAS = 5             # camera indexes 0..4 are tried when you press C
FRAME_W, FRAME_H = 1280, 720
FACE_MODEL = "yolo26s-face.pt"
FACE_MODEL_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolo26s-face.pt"
BODY_MODEL = "yolo26s.pt"
FACE_DETECT_CONF = 0.20      # lowest face confidence the model reports
BODY_DETECT_CONF = 0.40
MAX_PLAYERS = 4
PLAYER_LOST_AFTER = 1.0      # seconds without a body before a player is dropped
ROUND_TIME = 20.0
COUNTDOWN_TIME = 3.0
RESULT_TIME = 5.0
WIN_STREAK = 3.0             # seconds invisible in a row to win
IDLE_RETURN = 30.0           # seconds with nobody in view -> attract mode
INVITE_SWITCH = 4.0

TEXT = {
    "invites": [
        "Kan de computer jou zien?",
        "Kom dichterbij!",
        "Kun jij onzichtbaar worden voor AI?",
        "Pak een accessoire en verstop je!",
    ],
    "faces_today": "Vandaag {n} gezichten gezien",
    "invisible_today": "Vandaag {n} keer onzichtbaar",
    "stage_title": "Verstop je voor de AI!",
    "stage_hint": "Lichaam zichtbaar, gezicht niet = ONZICHTBAAR",
    "press_space": "Druk op SPATIE voor een nieuwe ronde",
    "get_ready": "Maak je klaar...",
    "invisible": "ONZICHTBAAR!",
    "won": "GEWONNEN!",
    "round_over": "Ronde voorbij!",
    "winners": "{n} winnaar(s)!",
    "no_winners": "Niemand was onzichtbaar... probeer iets anders!",
    "no_players": "Kom dichter bij het scherm om mee te doen",
    "face": "Gezicht",
    "prototype": "PROTOTYPE",
}

# colors (RGB)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (150, 150, 150)
GREEN = (60, 220, 120)
RED = (240, 70, 70)
PURPLE = (190, 90, 255)
YELLOW = (255, 210, 60)
PLAYER_COLORS = [(80, 180, 255), (255, 140, 60), (90, 230, 160), (255, 100, 200)]


def ensure_face_model():
    if os.path.exists(FACE_MODEL) and os.path.getsize(FACE_MODEL) > 1_000_000:
        return
    print(f"Downloading {FACE_MODEL} ...")
    r = requests.get(FACE_MODEL_URL, stream=True, timeout=60)
    r.raise_for_status()
    with open(FACE_MODEL, "wb") as f:
        for chunk in r.iter_content(1 << 16):
            f.write(chunk)


def tracked_boxes(result):
    """Return list of (track_id or None, x1, y1, x2, y2, conf)."""
    out = []
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
    for (x1, y1, x2, y2), c, tid in zip(xyxy, conf, ids):
        out.append((tid, float(x1), float(y1), float(x2), float(y2), float(c)))
    return out


class Player:
    def __init__(self, body_id, color):
        self.body_id = body_id
        self.color = color
        self.body = None
        self.face_conf = 0.0
        self.last_seen = time.time()
        self.streak = 0.0
        self.best_streak = 0.0
        self.won = False

    @property
    def invisible(self):
        return self.body is not None and self.face_conf < STATE["visible_thr"]


STATE = {
    "mode": "attract",          # attract | wait | countdown | playing | result
    "mode_start": time.time(),
    "debug": False,
    "player_min_face": 0.07,    # face width as fraction of frame width
    "visible_thr": 0.50,
    "face_ids_seen": set(),
    "invisible_today": 0,
    "last_person_seen": time.time(),
}


def set_mode(mode):
    STATE["mode"] = mode
    STATE["mode_start"] = time.time()


def main():
    ensure_face_model()
    face_model = YOLO(FACE_MODEL)
    body_model = YOLO(BODY_MODEL)

    def open_camera(index):
        c = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        if c.isOpened() and c.read()[0]:
            return c
        c.release()
        return None

    def next_camera(current):
        """Try the camera indexes after `current` and wrap around."""
        for step in range(1, MAX_CAMERAS + 1):
            idx = (current + step) % MAX_CAMERAS
            c = open_camera(idx)
            if c is not None:
                return idx, c
        return None, None

    cam_id = int(sys.argv[1]) if len(sys.argv) > 1 else CAMERA_ID
    cap = open_camera(cam_id)
    if cap is None:
        cam_id, cap = next_camera(cam_id)
    if cap is None:
        sys.exit("No working camera found.")
    STATE["toast"] = (f"Camera {cam_id}", time.time() + 3)

    pygame.init()
    screen = pygame.display.set_mode((FRAME_W, FRAME_H), pygame.RESIZABLE)
    pygame.display.set_caption("Evasion prototype")
    font_xl = pygame.font.SysFont("arial", 84, bold=True)
    font_l = pygame.font.SysFont("arial", 52, bold=True)
    font_m = pygame.font.SysFont("arial", 32, bold=True)
    font_s = pygame.font.SysFont("consolas", 18)
    font_xxl = pygame.font.SysFont("arial", 220, bold=True)
    clock = pygame.time.Clock()

    qr = None
    if os.path.exists("images/qr-code.png"):
        qr = pygame.transform.smoothscale(pygame.image.load("images/qr-code.png"), (110, 110))

    players = {}         # body_id -> Player
    particles = []       # confetti: [x, y, vx, vy, color, life]
    fullscreen = False
    t_prev = time.time()
    timings = {"face_ms": 0, "body_ms": 0}

    while True:
        now = time.time()
        dt = now - t_prev
        t_prev = now

        # --- input ---
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                cap.release(); pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_q:
                    cap.release(); pygame.quit(); sys.exit()
                elif k == pygame.K_SPACE and STATE["mode"] in ("attract", "wait", "result"):
                    for p in players.values():
                        p.streak = p.best_streak = 0.0
                        p.won = False
                    set_mode("countdown")
                elif k == pygame.K_ESCAPE:
                    set_mode("attract")
                elif k == pygame.K_d:
                    STATE["debug"] = not STATE["debug"]
                elif k == pygame.K_c:
                    cap.release()
                    new_id, new_cap = next_camera(cam_id)
                    if new_cap is None:  # nothing else works: reopen the old one
                        new_id, new_cap = cam_id, open_camera(cam_id)
                    cam_id, cap = new_id, new_cap
                    players.clear()
                    STATE["toast"] = (f"Camera {cam_id}", time.time() + 3)
                elif k == pygame.K_f:
                    fullscreen = not fullscreen
                    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) if fullscreen \
                        else pygame.display.set_mode((FRAME_W, FRAME_H), pygame.RESIZABLE)
                elif k == pygame.K_LEFTBRACKET:
                    STATE["player_min_face"] = max(0.02, STATE["player_min_face"] - 0.01)
                elif k == pygame.K_RIGHTBRACKET:
                    STATE["player_min_face"] = min(0.30, STATE["player_min_face"] + 0.01)
                elif k == pygame.K_MINUS:
                    STATE["visible_thr"] = max(0.20, STATE["visible_thr"] - 0.05)
                elif k == pygame.K_EQUALS:
                    STATE["visible_thr"] = min(0.90, STATE["visible_thr"] + 0.05)

        # --- camera + models ---
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)  # mirror view feels natural to people
        fh, fw = frame.shape[:2]

        t0 = time.time()
        faces = tracked_boxes(face_model.track(frame, persist=True, verbose=False, conf=FACE_DETECT_CONF)[0])
        t1 = time.time()
        bodies = tracked_boxes(body_model.track(frame, persist=True, verbose=False, conf=BODY_DETECT_CONF, classes=[0])[0])
        t2 = time.time()
        timings["face_ms"] = (t1 - t0) * 1000
        timings["body_ms"] = (t2 - t1) * 1000

        for f in faces:
            if f[0] is not None and f[5] >= STATE["visible_thr"]:
                STATE["face_ids_seen"].add(f[0])
        if bodies or faces:
            STATE["last_person_seen"] = now

        # --- match each face to the smallest body box that contains its center ---
        body_face = {}   # body_id -> (face box, conf)
        face_owner = {}  # face index -> body_id
        for i, (_, x1, y1, x2, y2, c) in enumerate(faces):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            best, best_area = None, None
            for (bid, bx1, by1, bx2, by2, _) in bodies:
                if bid is None:
                    continue
                if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                    area = (bx2 - bx1) * (by2 - by1)
                    if best is None or area < best_area:
                        best, best_area = bid, area
            if best is not None and (best not in body_face or c > body_face[best][1]):
                body_face[best] = ((x1, y1, x2, y2), c)
                face_owner[i] = best

        # --- update players: a body becomes a player when its face is large enough ---
        body_by_id = {b[0]: b for b in bodies if b[0] is not None}
        for bid, ((x1, _, x2, _), c) in body_face.items():
            big = (x2 - x1) / fw >= STATE["player_min_face"]
            if bid not in players and big and c >= STATE["visible_thr"] and len(players) < MAX_PLAYERS:
                used = {p.color for p in players.values()}
                color = next((col for col in PLAYER_COLORS if col not in used), PLAYER_COLORS[0])
                players[bid] = Player(bid, color)
        for bid, p in list(players.items()):
            b = body_by_id.get(bid)
            if b is not None:
                p.body = b[1:5]
                p.face_conf = body_face.get(bid, (None, 0.0))[1]
                p.last_seen = now
            else:
                p.body = None
                if now - p.last_seen > PLAYER_LOST_AFTER:
                    del players[bid]

        # --- game state machine ---
        mode = STATE["mode"]
        elapsed = now - STATE["mode_start"]
        if mode != "attract" and now - STATE["last_person_seen"] > IDLE_RETURN:
            set_mode("attract")
        elif mode == "countdown" and elapsed >= COUNTDOWN_TIME:
            set_mode("playing")
        elif mode == "playing":
            for p in players.values():
                if p.invisible:
                    p.streak += dt
                    p.best_streak = max(p.best_streak, p.streak)
                    if p.streak >= WIN_STREAK and not p.won:
                        p.won = True
                        STATE["invisible_today"] += 1
                        x1, y1, x2, y2 = p.body
                        for _ in range(80):
                            particles.append([(x1 + x2) / 2, y1 + 40, random.uniform(-9, 9),
                                              random.uniform(-14, -2), random.choice(PLAYER_COLORS + [YELLOW]), 1.6])
                else:
                    p.streak = 0.0
            if elapsed >= ROUND_TIME:
                set_mode("result")
        elif mode == "result" and elapsed >= RESULT_TIME:
            set_mode("wait")
        mode = STATE["mode"]
        elapsed = now - STATE["mode_start"]

        # --- draw video (scaled to window) ---
        sw, sh = screen.get_size()
        sx, sy = sw / fw, sh / fh
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(rgb.tobytes(), (fw, fh), "RGB")
        screen.blit(pygame.transform.scale(surf, (sw, sh)), (0, 0))

        def rect(x1, y1, x2, y2):
            return pygame.Rect(int(x1 * sx), int(y1 * sy), int((x2 - x1) * sx), int((y2 - y1) * sy))

        def text(s, font, color, center=None, topleft=None, shadow=True):
            img = font.render(s, True, color)
            r = img.get_rect(center=center) if center else img.get_rect(topleft=topleft)
            if shadow:
                screen.blit(font.render(s, True, BLACK), r.move(3, 3))
            screen.blit(img, r)
            return r

        def banner(y, h, alpha=150):
            band = pygame.Surface((sw, h), pygame.SRCALPHA)
            band.fill((0, 0, 0, alpha))
            screen.blit(band, (0, y))

        player_face_idx = {i for i, bid in face_owner.items() if bid in players}

        # faces: grey = passer-by, colored = player (only outside the game, or non-players)
        for i, (_, x1, y1, x2, y2, c) in enumerate(faces):
            if mode in ("countdown", "playing", "result") and i in player_face_idx:
                continue
            big = (x2 - x1) / fw >= STATE["player_min_face"]
            if c < STATE["visible_thr"]:
                continue
            col = GREEN if big else GREY
            r = rect(x1, y1, x2, y2)
            pygame.draw.rect(screen, col, r, 4 if big else 2, border_radius=8)
            if big:
                text(f"{TEXT['face']} {c:.0%}", font_m, col, topleft=(r.x, r.y - 40))

        # players during the game
        if mode in ("countdown", "playing", "result", "wait"):
            for p in players.values():
                if p.body is None:
                    continue
                x1, y1, x2, y2 = p.body
                r = rect(x1, y1, x2, y2)
                if p.invisible and mode == "playing":
                    pulse = 6 + int(4 * math.sin(now * 10))
                    pygame.draw.rect(screen, PURPLE, r, pulse, border_radius=14)
                    text(TEXT["invisible"], font_l, PURPLE, center=(r.centerx, r.y + 60))
                    # streak progress
                    frac = min(1.0, p.streak / WIN_STREAK)
                    pygame.draw.rect(screen, WHITE, (r.centerx - 120, r.y + 100, 240, 22), 3, border_radius=10)
                    pygame.draw.rect(screen, PURPLE, (r.centerx - 117, r.y + 103, int(234 * frac), 16), border_radius=8)
                else:
                    pygame.draw.rect(screen, p.color, r, 4, border_radius=14)
                if p.won:
                    text(TEXT["won"], font_l, YELLOW, center=(r.centerx, r.y + 160))
                # visibility bar: face confidence, green -> red
                bw = max(160, r.w - 40)
                bx, by = r.centerx - bw // 2, max(10, r.y - 46)
                fc = p.face_conf
                col = (int(RED[0] + (GREEN[0] - RED[0]) * fc), int(RED[1] + (GREEN[1] - RED[1]) * fc),
                       int(RED[2] + (GREEN[2] - RED[2]) * fc))
                pygame.draw.rect(screen, BLACK, (bx - 3, by - 3, bw + 6, 34), border_radius=10)
                pygame.draw.rect(screen, col, (bx, by, int(bw * fc), 28), border_radius=8)
                tx = bx + int(bw * STATE["visible_thr"])
                pygame.draw.line(screen, WHITE, (tx, by - 6), (tx, by + 34), 3)
                text(f"{TEXT['face']} {fc:.0%}", font_s, WHITE, topleft=(bx + 8, by + 4), shadow=False)

        # confetti
        for pt in particles:
            pt[0] += pt[2]; pt[1] += pt[3]; pt[3] += 0.5; pt[5] -= dt
            pygame.draw.rect(screen, pt[4], (int(pt[0] * sx), int(pt[1] * sy), 10, 14))
        particles[:] = [pt for pt in particles if pt[5] > 0]

        # --- overlays per mode ---
        if mode == "attract":
            banner(sh - 170, 170)
            invite = TEXT["invites"][int(now / INVITE_SWITCH) % len(TEXT["invites"])]
            text(invite, font_xl, WHITE, center=(sw // 2, sh - 105))
            text(TEXT["faces_today"].format(n=len(STATE["face_ids_seen"])), font_m, YELLOW, center=(sw // 2, sh - 35))
        else:
            banner(0, 90)
            text(TEXT["stage_title"], font_l, WHITE, center=(sw // 2, 32))
            text(TEXT["stage_hint"], font_s, WHITE, center=(sw // 2, 72), shadow=False)
            banner(sh - 60, 60)
            text(TEXT["invisible_today"].format(n=STATE["invisible_today"]), font_m, YELLOW, center=(sw // 2, sh - 30))
            if not players and mode != "result":
                text(TEXT["no_players"], font_l, WHITE, center=(sw // 2, sh // 2))
            if mode == "countdown":
                n = math.ceil(COUNTDOWN_TIME - elapsed)
                text(TEXT["get_ready"], font_l, WHITE, center=(sw // 2, sh // 2 - 90))
                text(str(n), font_xxl, YELLOW, center=(sw // 2, sh // 2 + 40))
            elif mode == "playing":
                left = max(0.0, ROUND_TIME - elapsed)
                text(f"{left:4.1f}", font_xl, YELLOW if left > 5 else RED, topleft=(sw - 230, 100))
            elif mode == "result":
                winners = sum(p.won for p in players.values())
                banner(sh // 2 - 90, 180, 190)
                text(TEXT["round_over"], font_xl, WHITE, center=(sw // 2, sh // 2 - 30))
                msg = TEXT["winners"].format(n=winners) if winners else TEXT["no_winners"]
                text(msg, font_m, YELLOW, center=(sw // 2, sh // 2 + 50))
            elif mode == "wait":
                text(TEXT["press_space"], font_m, WHITE, center=(sw // 2, sh // 2 + 200))

        if qr is not None:
            screen.blit(qr, (sw - 130, sh - 130 - (60 if mode != "attract" else 170)))
        text(TEXT["prototype"], font_s, RED, topleft=(10, sh - 24 if mode == "attract" else 96), shadow=False)
        toast = STATE.get("toast")
        if toast and now < toast[1]:
            text(toast[0], font_m, WHITE, center=(sw // 2, sh // 2 - 200))

        # --- debug overlay: full state ---
        if STATE["debug"]:
            lines = [
                f"fps {clock.get_fps():5.1f}  face {timings['face_ms']:5.1f}ms  body {timings['body_ms']:5.1f}ms",
                f"mode {mode}  t={elapsed:5.1f}s  camera {cam_id}",
                f"player_min_face [ ] = {STATE['player_min_face']:.2f}  visible_thr - = = {STATE['visible_thr']:.2f}",
                f"faces {len(faces)}  bodies {len(bodies)}  players {len(players)}",
            ]
            for p in players.values():
                lines.append(f"  P{p.body_id}: body={'yes' if p.body else 'NO '} face={p.face_conf:.2f} "
                             f"inv={p.invisible!s:5} streak={p.streak:4.1f} best={p.best_streak:4.1f} won={p.won}")
            for (bid, x1, y1, x2, y2, c) in bodies:
                pygame.draw.rect(screen, YELLOW, rect(x1, y1, x2, y2), 1)
                text(f"body {bid} {c:.2f}", font_s, YELLOW, topleft=(int(x1 * sx) + 4, int(y2 * sy) - 22), shadow=False)
            for (fid, x1, y1, x2, y2, c) in faces:
                pygame.draw.rect(screen, WHITE, rect(x1, y1, x2, y2), 1)
                text(f"f{fid} {c:.2f} w={(x2 - x1) / fw:.2f}", font_s, WHITE,
                     topleft=(int(x1 * sx), int(y2 * sy) + 2), shadow=False)
            pad = pygame.Surface((760, 22 * len(lines) + 12), pygame.SRCALPHA)
            pad.fill((0, 0, 0, 190))
            screen.blit(pad, (10, 110))
            for j, line in enumerate(lines):
                text(line, font_s, GREEN, topleft=(18, 116 + 22 * j), shadow=False)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
