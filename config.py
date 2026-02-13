# --- SYSTEM & GUI ---
GUI_TITLE = "Control Panel"
GUI_SIZE = "400x550"
WINDOW_NAME = "Live Detection Stream"
GUI_UPDATE_DELAY = 10  # Milliseconds between loop updates
GUI_TEXT = "Adversarial Demo"
GUI_TEXT_FONT = ("Arial", 16, "bold")
GUI_STATUS_LBL_COLOR = "blue"
GUI_SUGGESTION_LBL_COLOR = "gray"

# --- CAMERA ---
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
# Monitor Position: Move window to second screen (x=1920)
WINDOW_POS_X = 1920
WINDOW_POS_Y = 0

# --- RECORDING ---
RECORDING_FOLDER = "recordings"
RECORDING_FPS = 20.0
RECORDING_EXT = ".mp4"
# Codecs: Try H.264 ('avc1') for WhatsApp compatibility, fallback to 'mp4v'
CODEC_PRIMARY = 'avc1'
CODEC_FALLBACK = 'mp4v'

# --- AI MODELS ---
MODEL_OBJ_FILENAME = "yolov8n.pt"
MODEL_FACE_FILENAME = "yolov8n-face.pt"
# If face model is missing, download from here:
MODEL_FACE_URL = "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt"

# --- DETECTION SETTINGS ---
CONF_THRESHOLD_FACE = 0.5   # Confidence needed to show a face
CONF_THRESHOLD_OBJ = 0.4    # Confidence needed to show an object

# --- VISUALIZATION (BGR Colors) ---
COLOR_FACE_NORMAL = (0, 255, 0)    # Green
COLOR_FACE_PANDA = (0, 0, 255)     # Red (Adversarial)
COLOR_OBJECT = (0, 255, 255)       # Yellow
COLOR_RECORD_DOT = (0, 0, 255)     # Red dot

# --- ADVERSARIAL PATCH ---
# List of files to look for (in order of priority)
PATCH_FILES = ["glasses.png", "patch.png", "glasses.jpg"]
# Scale: How wide glasses are relative to eye distance (2.5x is standard width)
GLASSES_SCALE = 2.5

