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
OBJECT_CONF = 0.45           # lowest object confidence the model reports
OBJECT_MIN_FRAMES = 5        # an object must be seen this many frames in a row before it is shown
CONF_SURE = 0.80             # stage 2 colors: green from here
CONF_UNSURE = 0.60           # yellow from here, orange below

OBJECT_MIN_CONF = 0.10       # weakest object guess the model reports (for game meters only)

# --- stage 2 games ---
FOOL_TARGETS = ["banana", "apple", "orange", "carrot", "clock", "tie", "teddy bear",
                "cup", "scissors", "umbrella", "book", "donut", "cat", "dog", "giraffe"]
FOOL_ROUND_TIME = 20.0
FOOL_SEEN_MIN = 0.20         # "the AI sees now" shows guesses from this confidence
CONFUSE_ROUND_TIME = 30.0
CONFUSE_MIN_CONF = 0.30      # the followed object needs at least this confidence
CONFUSE_STABLE = 3           # a new label must hold this many detection updates to count
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
WIN_STREAK = 3.0             # seconds invisible in a row to win

# --- stage 4: simulated attack ---
GLASSES_FILE = ROOT / "glasses.png"
GLASSES_SCALE = 2.5          # glasses width relative to eye distance

# --- overlays ---
QR_FILE = ROOT / "images" / "qr-code.png"
