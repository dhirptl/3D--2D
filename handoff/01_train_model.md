# Handoff: Fine-tune the detection model on new Football Videos-2 data

You are working in an NFL broadcast player-detection/tracking pipeline. Your job
is **only** model training. Two other agents are concurrently editing this same
repo (one on HUD filtering, one on tracker re-ID) — see "File ownership" below.
Stay strictly inside your lane to avoid clobbering their edits.

## Environment (read carefully — the path has a trailing space)

- Repo root: `/Users/dhirpatel/3D->2D ` (note the trailing space in the dir name)
- Branch: `zerodet-instrumentation`
- Python: use the venv interpreter **`.venv/bin/python3.12`** (Python 3.12.13;
  the system `python3` is 3.14 with no torch — do not use it)
- torch 2.12.0 with MPS is installed; device is Apple M4 Pro
- Run python with `PYTHONPATH` set to the repo root, e.g.:
  `cd "/Users/dhirpatel/3D->2D " && PYTHONPATH="$PWD" .venv/bin/python3.12 ...`

## Background / what's already done

- Current production model: `football_tracker_seg/run_v3-4_hardneg/weights/best.pt`
  (YOLO11s-seg, single class `Player`, set as `SEG_MODEL_PATH` in `src/config.py`).
- A new labeled dataset, `Football Videos-2/` (Roboflow YOLO export, 6 classes:
  ball/player/players/referee/team_A/team_B), was added.
- **It was already deduped against the existing training set.** 235 of its 743
  images were byte-identical to images already in `merged_football_dataset_v2_seg/`.
  The 508 genuinely-new images, with classes remapped to single-class `Player`
  (player/players/team_A/team_B → 0; ball/referee dropped), are staged at:
  `data/fv2_clean/{train,valid}/{images,labels}/` (506 train, 2 valid).
- A training script exists: `scripts/retrain.py`. It (1) runs SAM mask generation
  on `data/fv2_clean/train` to convert the bbox labels to polygon masks and merges
  them into `merged_football_dataset_v2_seg/`, then (2) fine-tunes from
  `run_v3-4_hardneg` for 30 epochs → outputs `football_tracker_seg/run_v3-5_fv2/`.

## Your task

1. **Sanity-check first.** Run a dry run and confirm the dataset counts look right
   (~6,139 existing train + 508 new):
   ```
   PYTHONPATH="$PWD" .venv/bin/python3.12 scripts/retrain.py --dry-run
   ```
2. **Smoke-test SAM on a few images before the full run.** SAM mask generation is
   the riskiest step (mask quality). Verify `src/generate_seg_labels.py` produces
   sane polygons on ~5 images (it writes previews to `outputs/mask_preview/`).
   Eyeball a couple previews before committing to all 508.
3. **Run the full pipeline** (SAM gen ~30–60 min, then training; 30 epochs on
   ~6,647 images at imgsz=1024 on MPS may take several hours — consider reducing
   to 10–15 epochs for a first pass since only 508 images are new):
   ```
   PYTHONPATH="$PWD" .venv/bin/python3.12 scripts/retrain.py
   ```
   (If SAM gen already ran, use `--skip-sam`.)
4. **Evaluate** — see the eval caveat below.

## ⚠️ Eval caveat (important)

Do **not** trust `data/football_eval/` (the FV2 test split): 28 of its 29 images
are in the training set, so any mAP from it measures memorization, not skill.

The only clean signal right now is the confidence sweep on `1minclip.mov` (the
model never trained on it). After training, run a zero-det/precision proxy sweep:
- Pattern to copy: `/tmp/conf_sweep_hardneg.py` (or `src/eval_broadcast.py`).
- Compare new `run_v3-5_fv2` vs `run_v3-4_hardneg` at conf=0.10.
- Success = zero_det_rate ≤ 6.7% (no regression) and fp_rate ≤ 0.9% (ideally
  precision improves; hard-negative training already cut FPs).

A truly clean labeled eval set still needs to be built (Roboflow-label ~150
frames from 1minclip that are NOT in training) — out of scope for you, but note
it in your final summary.

## When done

- Update `src/config.py` line:
  `SEG_MODEL_PATH = ROOT / "football_tracker_seg" / "run_v3-5_fv2" / "weights" / "best.pt"`
  **only if** the sweep shows no regression. Commit that one-line change by itself
  with the before/after sweep numbers in the message.
- If the new model is worse, leave `SEG_MODEL_PATH` on `run_v3-4_hardneg` and
  report the numbers.

## File ownership (do NOT edit — other agents own these)

- `src/post_process.py` and `src/detection.py` — HUD-filter agent owns them.
- `configs/bytetrack.yaml` and tracker code — tracker re-ID agent owns it.
- You may touch: `data/`, `merged_football_dataset_v2_seg/`,
  `football_tracker_seg/`, and the single `SEG_MODEL_PATH` line in `src/config.py`.
- `merged_football_dataset_v2_seg/`, `data/fv2_clean/`, and `football_tracker_seg/`
  run dirs are gitignored — the dataset merge won't dirty git.

## Constraints

- One concern per commit. Don't bundle the model swap with anything else.
- Don't push; commits stay local until all three workstreams are validated.
