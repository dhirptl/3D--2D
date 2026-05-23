# Football Player Detection, Tracking & Team Classification

YOLOv11n-Seg + BoT-SORT (Re-ID) + automatic team color classification (WARMUP → LOCKED).

Color pipeline: mask isolate → turf subtract (green-safeguard) → YOLO-Pose torso polygon (or aspect-aware upper body) → LAB L-channel CLAHE → **4D LAB chrominance** (`lab4d_v1`) → KMeans **k=2** after pre-filtering. Playable area uses **homography** from yard lines when valid, with HSV fallback.

## Setup

```bash
cd "/Users/dhirpatel/3D->2D "
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
# or: pip install -r requirements.txt
```

Run scripts from the project root using the package module path:

```bash
python -m src.run_tracker --source ...
```

## Full pipeline

```bash
# 1. Bbox dataset
python -m src.prepare_dataset

# 2. SAM polygon labels (sam_b.pt) + QA
python -m src.generate_seg_labels
# Optional: --resume to skip existing labels; --preview-scale 0.5 for smaller previews

python -m src.prepare_seg_dataset

# 3. Train + Gate 2
python -m src.train_football_seg --epochs 100 --freeze 10
python -m src.validate_seg   # mask mAP50 >= 0.75, recall >= 0.70

# 4. Run on video
python -m src.run_tracker \
  --source outputs/arz_nyj_1min.mov \
  --out outputs/arz_nyj_1min_teams_tdd.mp4 \
  --save-calibration outputs/arz_nyj_cal_lab4d.pkl \
  --dump-detections outputs/arz_nyj_dets_v3.csv

# 5. Metrics
python -m src.eval_clip --source outputs/arz_nyj_1min.mov
```

## CLI flags

| Flag | Purpose |
|------|---------|
| `--no-team` | Detection/tracking only |
| `--no-hud-filter` | Disable HUD zone filter |
| `--filter-field` | Optional HSV foot-region filter |
| `--show-masks` | Overlay seg masks (ROI blend) |
| `--load-calibration` | Reuse team centroids (`feature_version=lab4d_v1` pickle only) |
| `--save-calibration` | Save centroids after LOCKED |
| `--tracker botsort\|bytetrack` | Tracker config (default: `botsort` with Re-ID) |
| `--pose-every N` | YOLO-Pose every N frames (default 1) |
| `--no-pose` | Disable pose torso sampling |
| `--dump-detections` | Stream CSV (includes `team_label`, distances) |
| `--debug-teams` | Show dist0/dist1 and warmup reject reasons on HUD |
| `--retina-masks` | Higher-res seg masks (auto-disabled if &gt;40ms/frame unless `--force-retina`) |
| `--force-retina` | Keep retina masks even when benchmark is slow |
| `--detect-every N` | Run YOLO every N frames (default 1); off-frames skip color extraction |
| `--codec avc1` | Output codec (`mp4v` fallback) |
| `--pipeline-threads N` | Read-ahead decode threads (default 4) |
| `--no-prefetch` | Disable async frame prefetch |
| `--hue-weight` | Deprecated (LAB features); ignored |

SAM label generation:

| Flag | Purpose |
|------|---------|
| `--resume` | Skip images with existing seg labels |
| `--preview-scale` | Scale preview JPEGs (e.g. 0.5) |
| `--workers` | Parallel copy workers (default 4) |

Training:

| Flag | Purpose |
|------|---------|
| `--epochs` | Training epochs (default 100) |
| `--freeze` | Frozen backbone layers (default 10; use 0 for full fine-tune) |

## Team tuning (`src/config.py`)

| Constant | Default | Effect |
|----------|---------|--------|
| `QUALITY_FRAMES` | 50 | Warmup vectors before calibration |
| `WARMUP_MIN_UNIQUE_TRACKS` | 12 | Distinct track IDs required |
| `WARMUP_MAX_PER_TRACK` | 3 | Cap samples per track |
| `FIELD_MASK_MIN_FRAC` | 0.25 | Mask overlap with turf for warmup |
| `MINIMUM_CENTROID_DISTANCE` | 1.5 | Min cluster separation to lock |
| `UPPER_BODY_FRAC` | 0.30 | Top fraction of mask for color stats |
| `YARD_LINE_SAT_MAX` / `YARD_LINE_VAL_MIN` | 60 / 180 | Yard-line pixels in turf mask |
| `MIN_UPPER_MASK_PIXELS` | 50 | Min upper-body pixels after turf subtract |
| `FEATURE_VERSION` | `lab4d_v1` | Calibration pickle schema |
| `WARMUP_FRAME_TIMEOUT` | 1500 | Force calibration attempt after this many frames |
| `KMEANS_RANDOM_STATE` | 42 | Reproducible KMeans clustering |

**Calibration pickles** using `6d_v1` or older are incompatible — delete the `.pkl` and re-run with `--save-calibration`.

### Ground-truth evaluation

Add rows to `annotations/sample_labels.csv` (`frame`, `track_id`, `correct_team`), then:

```bash
python -m src.eval_clip --source outputs/arz_nyj_1min.mov --annotations annotations/sample_labels.csv
```

## Troubleshooting

- **Grey boxes only:** Console prints `[warmup] rejection breakdown` every 150 frames (reasons: `low_conf`, `small_mask`, `few_upper_pixels`, `not_on_field`). Warmup auto-relaxes track threshold at 600/1200 frames and forces calibration after 1500. If you see `WARNING: Team colors may be too similar`, both teams may be indistinguishable by color.
- **Grey after camera cuts:** Cuts are detected via frame MAD; teams re-label directly for 60 frames without waiting for voter history.
- **Late LOCKED:** Low detections per frame — run `eval_clip` and check `locked_frame_index` (target &lt; 900 frames ≈ 30s).
- **Few players detected:** Run `validate_seg`; if Gate 2 fails, regenerate SAM labels or retrain.
- **Wrong turf filtering:** Auto HSV runs on first `FIELD_HSV_AUTO_FRAMES` frames; tune `FIELD_HSV_*` in config if needed.
- **Video won't play:** Install ffmpeg and try `--codec mp4v`.

## Tests

```bash
python tests/test_optimizations.py
```

## Models

| Model | Path |
|-------|------|
| Seg (primary) | `football_tracker_seg/run_v2/weights/best.pt` |
