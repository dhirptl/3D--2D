# Football Player Detection, Tracking & Team Classification

YOLOv11n-Seg + ByteTrack (default) + automatic team color classification (WARMUP → LOCKED).

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
# 1. Merge bbox sources into merged_football_dataset_v2
python -m src.prepare_merged_dataset_v2
# Optional:
# python -m src.mine_hard_negatives --source outputs/arz_nyj_1min.mov --split train
# python -m src.prepare_merged_dataset_v2

# 3. SAM polygon labels (sam_l.pt) + QA
python -m src.generate_seg_labels
# Optional: --resume to skip existing labels; --preview-scale 0.5 for smaller previews

python -m src.prepare_seg_dataset

# 4. Train + Gate 2
python -m src.train_football_seg --epochs 100 --freeze 10
python -m src.validate_seg

# 5. Optional helmet verification model
python -m src.prepare_helmet_dataset
python -m src.train_helmet_detector --epochs 50

# 6. Run on video
python -m src.run_tracker \
  --source outputs/arz_nyj_1min.mov \
  --out outputs/arz_nyj_1min_teams_tdd.mp4 \
  --save-calibration outputs/arz_nyj_cal_lab4d.pkl \
  --dump-detections outputs/arz_nyj_dets_v3.csv

# 7. Metrics
python -m src.eval_clip --source outputs/arz_nyj_1min.mov
python -m src.sweep_predict --source outputs/arz_nyj_1min.mov --max-frames 300
```

Dataset YAMLs (do not use root `data.yaml` for training):
- Segmentation: `merged_football_dataset_v2_seg/data.yaml` via `SEG_DATASET_YAML` in `src/config.py`
- Helmet: `football_dataset_helmet/data.yaml` via `HELMET_DATASET_YAML`
- Legacy bbox detector weights only: `football_tracker/run_v1/weights/best.pt` (`MODEL_PATH`)

## North-star release rubric

Use one representative clip with known on-field player density (recommended: a full play sequence, not warmup footage). A model/tracker setting is release-candidate only if all checks pass:

| Dimension | Metric source | Target |
|------|---------|--------|
| Detection coverage | `eval_clip` + manual spot-check | `avg_detections >= 18` and `zero_det_frames <= 5%` |
| Team stability | `eval_clip` (`locked_pct`, `label_flip_rate`, `locked_frame_index`) | `locked_pct >= 85%`, `label_flip_rate <= 0.10`, and `locked_frame_index < 900` |
| Team correctness | `eval_clip --annotations ...` | `team_accuracy.pct >= 90%` (when labels exist) |
| Runtime | `run_tracker` wall-clock on target machine | `>= 10 FPS` end-to-end for chosen quality mode |

Guidance:
- Treat `PLAYER_PREDICT_MAX_DET` as a safety bound, not a tuning floor; keep it high enough to avoid clipping dense frames.
- Only compare tracker variants or threshold sweeps after confirming dataset parity and passing the detection coverage gate above.

## CLI flags

| Flag | Purpose |
|------|---------|
| `--no-team` | Detection/tracking only |
| `--no-hud-filter` | Disable HUD zone filter |
| `--filter-field` | Optional playable-area foot-region filter (homography-backed with HSV fallback) |
| `--show-masks` | Overlay seg masks (ROI blend) |
| `--load-calibration` | Reuse team centroids (`feature_version=lab4d_v1` pickle only) |
| `--save-calibration` | Save centroids after LOCKED |
| `--tracker botsort\|bytetrack` | Tracker config (default: `bytetrack`) |
| `--save-kmeans-crops` | Dump torso-masked BGR tiles used for KMeans features |
| `--debug-crops-dir PATH` | Output dir for crop debug (default: `outputs/debug_crops`) |
| `--debug-crops-max-frames N` | Stop saving debug crops after frame N |
| `--pose-every N` | YOLO-Pose every N frames (default 1) |
| `--no-pose` | Disable pose torso sampling |
| `--dump-detections` | Stream CSV (includes `team_label`, distances) |
| `--debug-teams` | Show dist0/dist1 and warmup reject reasons on HUD |
| `--retina-masks` | Higher-res seg masks (auto-disabled if &gt;40ms/frame unless `--force-retina`) |
| `--force-retina` | Keep retina masks even when benchmark is slow |
| `--detect-every N` | Run YOLO every N frames (default 1); off-frames skip color extraction |
| `--player-conf FLOAT` | Player detector confidence threshold |
| `--player-iou FLOAT` | Player detector NMS IoU threshold |
| `--player-imgsz INT` | Player detector inference image size |
| `--player-max-det INT` | Player detector max detections per frame |
| `--codec avc1` | Output codec (`mp4v` fallback) |
| `--pipeline-threads N` | Read-ahead decode threads (default 4) |
| `--no-prefetch` | Disable async frame prefetch |
| `--hue-weight` | Deprecated (LAB features); ignored |
| `--no-helmet` | Disable default helmet verification |
| `--helmet-model PATH` | Override helmet detector weights |
| `--helmet-conf FLOAT` | Helmet detector confidence threshold |
| `--helmet-every N` | Run helmet verification every N detection frames |
| `--no-helmet-gate` | Disable temporal smoothing for helmet verification |
| `--helmet-gate-grace N` | Allow N initial frames before helmet confirmation is required |

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

Stage 1 training defaults now target:

- merged seg dataset: `merged_football_dataset_v2_seg/data.yaml`
- SAM labeler: `sam_l.pt`
- base model: `yolo11s-seg.pt`
- `mask_ratio=2`
- broadcast-oriented augments: `degrees=5`, `erasing=0.3`, `copy_paste=0.1`

## Helmet verification

Train the optional helmet model from the NFL Health & Safety dataset:

```bash
python -m src.prepare_helmet_dataset
python -m src.train_helmet_detector --epochs 50
```

Helmet verification is on by default. Run tracking with field filtering enabled:

```bash
python -m src.run_tracker \
  --source outputs/arz_nyj_1min.mov \
  --out outputs/arz_nyj_1min_teams_helmet.mp4 \
  --filter-field
```

Notes:

- `Helmet-Sideline` labels are excluded during dataset prep so the detector does not learn sideline helmet clusters as positives.
- `--filter-field` is recommended together with default helmet verification, because sideline staff or benched players may still wear helmets.
- Use `--no-helmet` when you explicitly want to evaluate the old behavior without helmet gating.
- Set `--helmet-gate-grace 0` for stricter offline analysis where you do not want tracks rendered before confirmation.

## Team tuning (`src/config.py`)

| Constant | Default | Effect |
|----------|---------|--------|
| `PLAYER_PREDICT_CONF` | 0.40 | YOLO player confidence (raise reduces sideline ghosts; lower with `--player-conf` if coverage drops) |
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

For inference calibration runs, `eval_clip` also accepts `--player-conf`, `--player-iou`, `--player-imgsz`, and `--player-max-det`.

### Gate 2 validation

`validate_seg` now reports and gates on:

- box recall
- mask mAP@0.50
- mask mAP@0.75
- mask recall

It also prints size diagnostics (`small`, `medium`, `large`) as box recall@0.5 by GT area so you can see whether background-player failures are concentrated in the smallest bucket.

### Hard negatives

Use `mine_hard_negatives` to collect crowd / bench / sideline false-positive frames into `football_dataset/hard_negatives/`, then re-run `prepare_merged_dataset_v2` before regenerating seg labels:

```bash
python -m src.mine_hard_negatives --source outputs/arz_nyj_1min.mov --split train
python -m src.prepare_merged_dataset_v2
python -m src.generate_seg_labels
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
pytest tests/ -m "not slow" -q
# legacy: python tests/test_optimizations.py
```

## Models

| Model | Path |
|-------|------|
| Seg (primary) | `football_tracker_seg/run_v2/weights/best.pt` |
| Helmet (optional) | `football_tracker_helmet/run_v1/weights/best.pt` |
