"""Settings for the stage show. Values marked (live) can be tuned with keys in debug mode."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- camera and window ---
CAMERA_ID = 0                # override on the command line: python -m show 1
MAX_CAMERAS = 5              # camera indexes tried when switching with C
FRAME_W, FRAME_H = 1280, 720
START_FULLSCREEN = False
RENDER_W, RENDER_H = 1280, 720  # the show is drawn at this size; the graphics card scales it to the screen
SHOW_FPS = 30                # screen images per second; more only takes time from the AI
SHOW_DISPLAY = None          # screen number for the show; None = largest landscape screen that is not the laptop
PANEL_DISPLAY = 0            # screen number of the laptop (operator panel)

# --- models ---
FACE_MODEL = "yolo26s-face.pt"
FACE_MODEL_URL = "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolo26s-face.pt"
OBJECT_MODEL = "yolo26s.pt"  # downloaded automatically by ultralytics
SEG_MODEL = "yolo26s-seg.pt"  # object shapes for the silhouette game; downloaded automatically
SEG_CONF = 0.25

# --- detection ---
FACE_DETECT_CONF = 0.20      # lowest face confidence the model reports
FACE_VISIBLE_CONF = 0.50     # (live) a face counts as seen above this
SHOW_SCORES = True           # (live, V) face scores on screen; off = stage 3 shows them only during a round
OBJECT_CONF = 0.45           # lowest object confidence the model reports
OBJECT_MIN_FRAMES = 5        # an object must be seen this many frames in a row before it is shown
CONF_SURE = 0.80             # stage 2 colors: green from here
CONF_UNSURE = 0.60           # yellow from here, orange below

OBJECT_MIN_CONF = 0.10       # weakest object guess the model reports (for game meters only)
STATIC_OBJECTS = {"tv", "couch", "chair", "bed", "dining table", "refrigerator", "oven", "sink",
                  "toilet", "potted plant", "bench", "microwave", "car", "bus", "truck", "train",
                  "airplane", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter"}
                             # objects nobody holds up: never shown in stage 2, never picked by a game
HELD_MAX_SIZE = 0.5          # a held object is at most this part of the player's body box area
STAGE2_MAX_OBJECTS = 5       # stage 2 free detection: at most this many object boxes (held ones first)

# --- stage 2 games ---
FOOL_TARGETS = ["banana", "apple", "orange", "carrot", "donut", "clock", "cup", "scissors",
                "teddy bear", "umbrella", "tie", "book"]
                             # fixed order; the game starts at the first one, T / Shift+T step through
FOOL_ROUND_TIME = 20.0
FOOL_SEEN_MIN = 0.20         # "the AI sees now" shows guesses from this confidence
CONFUSE_ROUND_TIME = 30.0
CONFUSE_MIN_CONF = 0.30      # the followed object needs at least this confidence
CONFUSE_STABLE = 3           # a new label must hold this many detection updates to count
CONFUSE_LOCK_IOU = 0.2       # lost tracker id: the object with this much box overlap is the same object
CONFUSE_MAX_CANDIDATES = 6   # numbered objects the operator can choose from before a round
HUNT_TARGETS = ["bottle", "cell phone", "backpack", "book", "cup", "handbag", "laptop",
                "umbrella", "chair", "clock", "scissors", "toothbrush"]
HUNT_ROUND_TIME = 45.0
HUNT_MIN_CONF = 0.60         # an object counts as found from this confidence ...
HUNT_STABLE = 3              # ... for this many detection updates in a row
HUNT_SKIP_AFTER = 10.0       # seconds before the game moves on to the next object by itself
LEADERBOARD_FILE = ROOT / "leaderboard.json"
ATTRACT_BOARD_EVERY = 20.0   # attract mode: show a leaderboard every ... seconds
ATTRACT_BOARD_TIME = 8.0     # ... for this long

# --- players ---
PLAYER_MIN_FACE = 0.07       # (live) face width / frame width to count as a player
MAX_PLAYERS = 4
PLAYER_LOST_AFTER = 1.0      # seconds without a body before a player is dropped

# --- attract mode ---
ATTRACT_MAX_BOXES = 12       # at most this many people get a box (the largest first)

# --- show flow ---
IDLE_RETURN = 30.0           # seconds with nobody in view -> attract mode
INVITE_SWITCH = 4.0
DEFAULT_LANG = "nl"

# --- stage 1: pixel grid ---
PIXEL_GRID = 20              # cells per side in the final number grid

# --- stage 3: evasion game ---
ROUND_TIME = 20.0
COUNTDOWN_TIME = 3.0
RESULT_TIME = 5.0
WIN_STREAK = 3.0             # seconds invisible in a row to win (again: a player can win more than once)
WIN_SHOW_TIME = 3.0          # seconds the "won" text shows after each win
EVASION_MODE = "hard"        # (live, M) "hard": modern face model (YOLO); "easy": classic Haar cascade (2001)
HAAR_FILE = "haarcascade_frontalface_default.xml"  # ships with OpenCV
HAAR_SCALE = 0.5             # the cascade runs on a smaller image: faster, close faces stay big enough
HAAR_MIN_HITS = 6            # (live) easy mode: a face counts as seen from this many overlapping hits
HAAR_HITS_FULL = 40          # hits that fill the bar on screen

# --- stage 4: simulated attack ---
GLASSES_FILE = ROOT / "assets" / "glasses.png"
GLASSES_SCALE = 2.5          # glasses width relative to eye distance

# --- overlays ---
HEADER_H = 44                # height of the stage title line at the top
SUBTITLE_TIME = 8.0          # the stage subtitle shows this long after a stage starts, then fades
QR_FILE = ROOT / "assets" / "qr-code.png"
