from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Detection / tracking ---
PLAYER_CLASS_ID = 0
PLAYER_IMGSZ = 736
PLAYER_PREDICT_CONF = 0.30
PLAYER_PREDICT_IOU = 0.65
PLAYER_PREDICT_MAX_DET = 30

HUD_TOP_PCT = 0.10
HUD_BOTTOM_PCT = 0.15

FIELD_HSV_HUE_LOW = 35
FIELD_HSV_HUE_HIGH = 85
FIELD_HSV_SAT_LOW = 40
FIELD_HSV_VAL_LOW = 40

# Detection model (legacy bbox)
MODEL_PATH = ROOT / "football_tracker" / "run_v1" / "weights" / "best.pt"

# Segmentation model (primary)
SEG_MODEL_PATH = ROOT / "football_tracker_seg" / "run_v1-2" / "weights" / "best.pt"
SEG_DATASET_YAML = ROOT / "football_dataset_seg" / "data.yaml"
SEG_DATASET_ROOT = ROOT / "football_dataset_seg"

TRACKER_CFG = ROOT / "configs" / "bytetrack.yaml"
DATASET_YAML = ROOT / "football_dataset" / "data.yaml"

# --- SAM label generation ---
SAM_MODEL = "sam_b.pt"
MASK_MIN_AREA_FLOOR = 300
MASK_MIN_AREA_BOX_FRAC = 0.15
MASK_MIN_BOX_IOU = 0.35
MASK_BOX_PAD_FRAC = 0.05
MASK_PREVIEW_COUNT = 50

# Gate 2 seg validation targets
GATE2_MASK_MAP50 = 0.75
GATE2_MASK_RECALL = 0.70

# --- Team classification (TDD Part 9) ---
WARMUP_CONF_MIN = 0.45
QUALITY_FRAMES = 80
W_H = 2.5
W_B = 0.5
TEAM_REF_S_MAX = 25
TEAM_REF_STD_V_MIN = 50
MINIMUM_CENTROID_DISTANCE = 1.5
MINIMUM_CENTROID_DISTANCE_FALLBACK = 1.0
CALIBRATION_FAIL_MAX = 3

FIELD_HSV_AUTO_FRAMES = 30

DEQUE_LENGTH = 20
CONFIDENCE_MARGIN = 1.5
SOFT_LOCK_THRESHOLD = 10
ESCAPE_THRESHOLD = 6
ESCAPE_MARGIN = 3.0

# Scaler fits raw 12D columns [1,2, 4,5, 7,8, 10,11] -> indices 0..7
SCALER_SV_COLS = [1, 2, 4, 5, 7, 8, 10, 11]
