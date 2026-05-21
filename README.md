# Football Player Detection, Tracking & Team Classification

YOLOv11n-Seg + ByteTrack + automatic team color classification (WARMUP → LOCKED).

## Setup

```bash
cd "/Users/dhirpatel/3D->2D "
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Full pipeline

```bash
# 1. Bbox dataset
python src/prepare_dataset.py

# 2. SAM polygon labels (sam_b.pt) + QA
python src/generate_seg_labels.py
# Review outputs/mask_preview/ and outputs/mask_qa/stats.json (target keep_rate > 85%)

python src/prepare_seg_dataset.py

# 3. Train + Gate 2
python src/train_football_seg.py
python src/validate_seg.py   # mask mAP50 >= 0.75, recall >= 0.70

# 4. Run on video
python src/run_tracker.py \
  --source outputs/arz_nyj_1min.mov \
  --out outputs/arz_nyj_1min_teams_v2.mp4 \
  --save-calibration outputs/arz_nyj_cal.pkl \
  --dump-detections outputs/arz_nyj_dets.csv

# 5. Metrics
python src/eval_clip.py --source outputs/arz_nyj_1min.mov
```

## CLI flags

| Flag | Purpose |
|------|---------|
| `--no-team` | Detection/tracking only |
| `--no-hud-filter` | Disable HUD zone filter |
| `--filter-field` | Optional HSV foot-region filter |
| `--show-masks` | Overlay seg masks |
| `--load-calibration` | Reuse team centroids |
| `--save-calibration` | Save centroids after LOCKED |
| `--dump-detections` | CSV per-frame stats |

## Troubleshooting

- **Grey boxes only:** Classifier stuck in WARMUP — check overlay counter and console for `CALIBRATION FAILED`. Lower `QUALITY_FRAMES` or `MINIMUM_CENTROID_DISTANCE` in `src/config.py`.
- **Few players detected:** Run `validate_seg.py`; if Gate 2 fails, regenerate SAM labels or retrain.
- **Wrong turf filtering:** Auto HSV runs on first 30 frames; tune `FIELD_HSV_*` in config if needed.

## Models

| Model | Path |
|-------|------|
| Seg (primary) | `football_tracker_seg/run_v1-2/weights/best.pt` |
| Detect (legacy) | `football_tracker/run_v1/weights/best.pt` |
