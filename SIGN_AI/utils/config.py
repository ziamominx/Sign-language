"""
Central configuration for the sign-language recognition system.

Every constant that affects the shape or meaning of the feature vector
lives here, so collect_data.py, train_model.py and test_model.py all
stay perfectly in sync. Never redefine these numbers elsewhere.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
DATASET_CSV = os.path.join(DATASET_DIR, "dataset.csv")
DATASET_METADATA_CSV = os.path.join(DATASET_DIR, "dataset_metadata.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_FILE_KERAS = os.path.join(MODELS_DIR, "sign_language_model.h5")
MODEL_FILE_SKLEARN = os.path.join(MODELS_DIR, "sign_language_model.pkl")
LABEL_ENCODER_FILE = os.path.join(MODELS_DIR, "label_encoder.pkl")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")
CONFIG_FILE = os.path.join(MODELS_DIR, "config.json")

# MediaPipe Tasks model bundles (auto-downloaded on first run -- see
# utils/landmarks.py). Cached here so later runs don't need internet.
MP_MODELS_DIR = os.path.join(PROJECT_ROOT, "mp_models")
HAND_LANDMARKER_MODEL_PATH = os.path.join(MP_MODELS_DIR, "hand_landmarker.task")
FACE_LANDMARKER_MODEL_PATH = os.path.join(MP_MODELS_DIR, "face_landmarker.task")
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# ---------------------------------------------------------------------------
# MediaPipe settings
# ---------------------------------------------------------------------------
MAX_NUM_HANDS = 2
HAND_DETECTION_CONFIDENCE = 0.6
HAND_TRACKING_CONFIDENCE = 0.5

FACE_DETECTION_CONFIDENCE = 0.6
FACE_TRACKING_CONFIDENCE = 0.5

NUM_HAND_LANDMARKS = 21  # fixed by MediaPipe Hands

# Landmark indices (within the 21 hand landmarks) used as reference points
# for computing hand-to-face distances.
HAND_REFERENCE_POINTS = {
    "wrist": 0,
    "thumb_tip": 4,
    "index_tip": 8,
}
# Landmarks used to estimate hand scale (for size-invariant normalization).
HAND_WRIST_IDX = 0
HAND_MIDDLE_MCP_IDX = 9  # base of the middle finger

# Named MediaPipe FaceMesh landmark indices we keep. Chosen to cover the
# forehead, chin, nose, eyes, mouth and cheeks -- the regions signers
# actually touch or reference.
FACE_LANDMARK_IDS = {
    "forehead": 10,
    "glabella": 9,          # between the eyebrows
    "nose_tip": 1,
    "nose_bottom": 2,
    "chin": 152,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_outer": 263,
    "right_eye_inner": 362,
    "left_cheek": 234,
    "right_cheek": 454,
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip": 13,
    "lower_lip": 14,
    "left_ear": 127,
    "right_ear": 356,
}
# Ordered list -- iteration order must stay fixed for a stable feature vector.
FACE_LANDMARK_NAMES = list(FACE_LANDMARK_IDS.keys())

# Subset of the face landmarks above used as reference points when computing
# hand-to-face distances (kept small and semantically meaningful).
FACE_REFERENCE_POINTS = [
    "forehead", "chin", "nose_tip",
    "left_eye_outer", "right_eye_outer", "mouth_center",
]

# ---------------------------------------------------------------------------
# Sequence / recording settings
# ---------------------------------------------------------------------------
RECORDING_DURATION_SECONDS = 5.0
TARGET_CAPTURE_FPS = 20          # frames we try to capture per second
MAX_SEQUENCE_LEN = 100           # hard cap on frames kept per sequence
MIN_SEQUENCE_LEN = 8             # sequences shorter than this are rejected

# Fixed sequence length the model is trained/tested on. Sequences are
# padded (by repeating the last frame) or truncated (uniform sampling)
# to this length.
MODEL_SEQUENCE_LEN = 60

# ---------------------------------------------------------------------------
# Real-time smoothing
# ---------------------------------------------------------------------------
PREDICTION_BUFFER_SIZE = 8
MIN_CONFIDENCE_THRESHOLD = 0.50
MIN_CONSECUTIVE_AGREEMENT = 5  # out of PREDICTION_BUFFER_SIZE

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MP_MODELS_DIR, exist_ok=True)
