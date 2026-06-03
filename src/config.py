from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Detection / tracking ---
PLAYER_CLASS_ID = 0
PLAYER_IMGSZ = 1280
PLAYER_PREDICT_CONF = 0.18
PLAYER_PREDICT_IOU = 0.45
PLAYER_PREDICT_MAX_DET = 100

HUD_TOP_PCT = 0.10
HUD_BOTTOM_PCT = 0.15
HUD_FIELD_TOP_MARGIN_PX = 20
HUD_FIELD_BOTTOM_MARGIN_PX = 35
FIELD_FOOT_MIN_FRAC = 0.02
FIELD_FOOT_MIN_FRAC_DEFAULT = 0.10

FIELD_HSV_HUE_LOW = 35
FIELD_HSV_HUE_HIGH = 85
FIELD_HSV_SAT_LOW = 40
FIELD_HSV_VAL_LOW = 40

# Yard lines (white/yellow markings) for turf mask
YARD_LINE_SAT_MAX = 60
YARD_LINE_VAL_MIN = 180

# TDD upper-body color sampling
UPPER_BODY_FRAC = 0.30
MIN_COLOR_PIXELS = 10
MIN_UPPER_MASK_PIXELS = 50

# Detection model (legacy bbox)
MODEL_PATH = ROOT / "football_tracker" / "run_v1" / "weights" / "best.pt"

# Segmentation model (primary)
SEG_MODEL_PATH = ROOT / "football_tracker_seg" / "run_v3-4_hardneg" / "weights" / "best.pt"
SEG_BBOX_SOURCE_ROOT = ROOT / "merged_football_dataset_v2"
SEG_DATASET_ROOT = ROOT / "merged_football_dataset_v2_seg"
SEG_DATASET_YAML = SEG_DATASET_ROOT / "data.yaml"
SEG_TRAIN_MODEL = "yolo11s-seg.pt"
SEG_MASK_RATIO = 4
SEG_TRAIN_DEGREES = 5.0
SEG_TRAIN_ERASING = 0.3
SEG_TRAIN_COPY_PASTE = 0.4
SEG_CLOSE_MOSAIC = 20

DATASET_YAML = ROOT / "football_dataset" / "data.yaml"
HELMET_DATASET_YAML = ROOT / "football_dataset_helmet" / "data.yaml"
HELMET_MODEL_PATH = ROOT / "football_tracker_helmet" / "run_v1" / "weights" / "best.pt"

# --- SAM label generation ---
SAM_MODEL = "sam_l.pt"
MASK_MIN_AREA_FLOOR = 400
MASK_MIN_AREA_BOX_FRAC = 0.20
MASK_MIN_BOX_IOU = 0.65
MASK_BOX_PAD_FRAC = 0.05
MASK_PREVIEW_COUNT = 50

# Gate 2 seg validation targets
GATE2_MASK_MAP50 = 0.75
GATE2_MASK_RECALL = 0.70
GATE2_MASK_MAP75 = 0.55
GATE2_BOX_RECALL = 0.75

# Clip evaluation north-star targets
EVAL_MIN_AVG_DETECTIONS = 18.0
EVAL_MAX_ZERO_DET_FRAC = 0.05
EVAL_MAX_ACTIONABLE_ZERO_DET_FRAC = 0.05  # HUD + formation-boundary only (decomposed)
EVAL_MIN_LOCKED_PCT = 85.0
EVAL_MAX_LABEL_FLIP_RATE = 0.10
EVAL_MAX_LOCKED_FRAME = 900
EVAL_MIN_TEAM_ACCURACY_PCT = 90.0
EVAL_MAX_TRACK_ID_CHANGE_RATE = 0.40

# --- Team classification (TDD Part 9) ---
WARMUP_CONF_MIN = 0.45
WARMUP_CONF_HELMET = 0.30
WARMUP_CONF_RELAXED = 0.38
WARMUP_CONF_RELAX_AFTER_FAILS = 2
QUALITY_FRAMES = 35
WARMUP_MAX_PER_TRACK = 3
WARMUP_MIN_UNIQUE_TRACKS = 8
FIELD_MASK_MIN_FRAC = 0.25
W_H = 2.5
MASK_ERODE_PX = 2
TEAM_REF_S_MAX = 25
TEAM_REF_STD_V_MIN = 50
TEAM_REF_MAX_MASK_AREA = 2500
TEAM_REF_MIN_ASPECT = 1.35
MINIMUM_CENTROID_DISTANCE = 1.5
MINIMUM_CENTROID_DISTANCE_FALLBACK = 1.0
CALIBRATION_FAIL_MAX = 3
CALIBRATION_VERSION = 2
FEATURE_VERSION = "lab6d_v2"

FIELD_HSV_AUTO_FRAMES = 30

DEQUE_LENGTH = 20
CONFIDENCE_MARGIN = 1.5
CONFIDENCE_MARGIN_MIN = 0.8
CONFIDENCE_MARGIN_FRAC = 0.12
WARMUP_CONFIDENCE_MARGIN = 1.0
WARMUP_SOFT_LOCK_THRESHOLD = 6
ESCAPE_MARGIN_FRAC = 0.25
ESCAPE_MARGIN_MIN = 2.0
SOFT_LOCK_THRESHOLD = 10
ESCAPE_THRESHOLD = 6
ESCAPE_MARGIN = 3.0

# Scaler fits all 6 feature dimensions (LAB AB + HSV hue)
SCALER_LAB_COLS = [0, 1, 2, 3, 4, 5]
SCALER_SV_COLS = SCALER_LAB_COLS  # backward-compat alias

# Pre-filter before KMeans k=2
PREFILTER_CHROMA_MAX = 18.0  # max |mean_A-128| and |mean_B-128| for neutral/ref
PREFILTER_FIELD_AB_DIST = 25.0  # drop samples near field AB centroid
TEAM_REF_AB_STD_MAX = 12.0  # low chrominance std = referee-like

# Green-safeguard turf subtract
GREEN_SAFEGUARD_HSV_FRAC = 0.55
GREEN_SAFEGUARD_AB_STD_MAX = 15.0

# Field homography registration
FIELD_REG_MIN_INLIERS = 4
FIELD_REG_MAX_REPROJ_ERR = 15.0
FIELD_REG_VALID_STREAK = 3
FIELD_REG_TEMPLATE_W = 1200
FIELD_REG_TEMPLATE_H = 533
FIELD_REG_UPDATE_STRIDE_STABLE = 15  # frames to skip per Hough+RANSAC when locked
FIELD_REG_STABLE_STREAK = 5          # valid_streak depth considered stable
FIELD_REG_MIN_PROJ_AREA_FRAC = 0.08  # reject collapsed template projection (gap 0.05–0.12 on 1minclip)

# Shared field ROI fractions (calibration, rolling update, green fraction)
FIELD_ROI_Y0_FRAC = 0.35
FIELD_ROI_Y1_FRAC = 0.85
FIELD_ROI_X0_FRAC = 0.10
FIELD_ROI_X1_FRAC = 0.90
FIELD_HSV_CALIB_PX_PER_FRAME = 2000  # pixels sampled per frame for HSV calibration

# Spatial prior calibration
SPATIAL_MIN_PLAYERS = 14
SPATIAL_MIN_GROUP_SIZE = 5
SPATIAL_CONF_MIN = 0.35
SPATIAL_MIN_REL_SEPARATION = 0.20
SPATIAL_MAX_SPREAD_FRAC = 0.60

# Pose model
POSE_MODEL_PATH = ROOT / "yolo11n-pose.pt"
POSE_EVERY_DEFAULT = 2
POSE_IOU_MATCH = 0.5

# Helmet verification
HELMET_CONF = 0.20
HELMET_HEAD_IOU = 0.12
HEAD_ROI_FRAC = 0.28
HELMET_EVERY_DEFAULT = 4
HELMET_GATE_WINDOW = 10
HELMET_GATE_REQUIRED = 1
HELMET_GATE_GRACE = 15
HELMET_GATE_SOFT_LOCK_STREAK = 7
HELMET_GATE_LOCKED_REQUIRED = 1

# Tracker (default ByteTrack; BoT-SORT available via --tracker botsort)
BOTSORT_CFG = ROOT / "configs" / "botsort.yaml"
BYTETRACK_CFG = ROOT / "configs" / "bytetrack.yaml"
TRACKER_CFG = BYTETRACK_CFG

# Feature debug crop dump (run_tracker --save-kmeans-crops)
DEBUG_CROP_DIR = ROOT / "outputs" / "debug_crops"

# Warmup timeout & progressive relaxation
WARMUP_FRAME_TIMEOUT = 1500
WARMUP_RELAX_AT_600 = 600
WARMUP_RELAX_AT_1200 = 1200
CALIBRATION_COOLDOWN_FRAMES = 150
CALIBRATION_MIN_NEW_SAMPLES = 15
WARMUP_TIMEOUT_MIN_VECTORS = 20

# Scaler sanity (legacy name; LAB uses B channel spread)
EXPECTED_S_MEAN_RANGE = (40, 180)
EXPECTED_B_MEAN_RANGE = (90, 170)

# Field mask cache & rolling HSV
FIELD_MASK_INTERVAL = 30
FIELD_HSV_UPDATE_INTERVAL = 300
FIELD_HSV_BLEND_ALPHA = 0.1
CAMERA_CUT_MAD_THRESHOLD = 45.0
CAMERA_CUT_CONSECUTIVE_FRAMES = 2
POST_CUT_FRAMES = 60

# Pile-up guard
PILE_UP_DISTANCE_PX = 80
PILE_UP_ESCAPE_THRESHOLD = 20
MASK_AREA_FREEZE_MULT = 2.0

# Similar-color warning
SIMILAR_COLOR_WARN_ATTEMPTS = 10
SIMILAR_COLOR_AVG_DIST_THRESHOLD = 0.8

# Bounding box EMA smoothing (0 = off, 1 = no smoothing)
BBOX_EMA_ALPHA = 0.6

# Duplicate suppression after tracking
DUPLICATE_SUPPRESS_IOU = 0.70

# Half-precision inference (FP16). Significant speedup on CUDA; variable on MPS.
PLAYER_PREDICT_HALF = False

# Inference
DETECT_EVERY_DEFAULT = 1
KMEANS_N_INIT = 10
KMEANS_RANDOM_STATE = 42

# Offline team assignment: re-cluster per camera segment only if luminance differs
OFFLINE_BRIGHTNESS_SPLIT_DELTA = 15.0  # gray mean 0-255 across cut segments
OFFLINE_BRIGHTNESS_LUMINANCE_STRIDE = 30  # sample every N frames per segment
FIELD_HSV_MAX_PIXELS = 50_000
WARMUP_VECTOR_CAP = QUALITY_FRAMES * 2
PIPELINE_THREADS_DEFAULT = 4
RETINA_BENCHMARK_FRAMES = 10
RETINA_MAX_MS_PER_FRAME = 40.0
