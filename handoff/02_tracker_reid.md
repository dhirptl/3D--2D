# Handoff: Reduce ID switches after formation changes (tracker re-ID)

You are working in an NFL broadcast player-detection/tracking pipeline. Your job
is **only** tracking / re-identification. Two other agents are concurrently
editing this same repo (one training the model, one fixing HUD filtering) — see
"File ownership" below. Stay strictly in your lane.

## Environment (the path has a trailing space)

- Repo root: `/Users/dhirpatel/3D->2D ` (trailing space is real)
- Branch: `zerodet-instrumentation`
- Python: `.venv/bin/python3.12` (3.12.13; system python3 is 3.14, no torch)
- Run with PYTHONPATH set to repo root:
  `cd "/Users/dhirpatel/3D->2D " && PYTHONPATH="$PWD" .venv/bin/python3.12 ...`

## The problem

Players get new track IDs after formation changes, occlusion piles (line of
scrimmage), and players leaving/re-entering frame. Baseline on the eval clip
(NOTE: this baseline is stale — measured at the old conf=0.40 with run_v3-3;
**re-baseline first**, see below):

| metric | baseline | meaning |
|--------|----------|---------|
| id_switches | 650 | track ID reassigned to a different player |
| fragmentation_rate | 134 | a single player's track broken into pieces |
| track_id_change_rate | 0.53 | fraction of frames where IDs churn |

## Key technical context

- Tracker is **ByteTrack** via ultralytics `model.track(..., persist=True)` in
  `src/detection.py` (~line 116). Config: **`configs/bytetrack.yaml`**:
  ```
  track_high_thresh: 0.25
  track_low_thresh: 0.08
  new_track_thresh: 0.25
  track_buffer: 90          # frames a lost track is kept for re-matching
  match_thresh: 0.65
  fuse_score: true
  ```
- **Critical insight:** ByteTrack is **motion/IoU-only — it has NO appearance
  model.** When a player is occluded in a pile and reappears displaced, IoU
  association fails and they get a new ID. True "re-identification after
  formation changes" fundamentally needs **appearance features**. Two paths:
  1. **BoT-SORT with ReID** — `configs/botsort.yaml` already exists; ultralytics
     BoT-SORT supports `with_reid: true` (appearance embeddings). Try enabling it
     and switching the default. This is the most direct fix for occlusion re-ID.
  2. **Custom post-association** — a re-ID layer that matches lost tracks to new
     detections by appearance (color histogram / embedding) after ByteTrack runs.
- `--tracker botsort` is already a CLI flag in `src/run_tracker.py` and
  `src/eval_clip.py`, so you can A/B ByteTrack vs BoT-SORT without code changes.

## Your task

1. **Re-baseline at current settings.** The model is now `run_v3-4_hardneg` at
   conf=0.10 (much higher recall than the stale baseline). Measure tracking
   metrics fresh with `src/eval_clip.py` on `1minclip.mov`:
   ```
   PYTHONPATH="$PWD" .venv/bin/python3.12 -m src.eval_clip \
       --source 1minclip.mov --tracker bytetrack
   ```
   It reports id_switches, fragmentation_rate, track_id_change_rate.
2. **A/B BoT-SORT with ReID** vs tuned ByteTrack. Run the same eval with
   `--tracker botsort`. Enable `with_reid: true` in `configs/botsort.yaml` and
   compare. Appearance ReID should specifically help post-occlusion re-id.
3. **Tune** whichever wins. For ByteTrack: `track_buffer` (longer keeps lost
   tracks alive across occlusion), `match_thresh`, `new_track_thresh` (higher =
   fewer spurious new IDs). For BoT-SORT: ReID weight, `proximity_thresh`,
   `appearance_thresh`.
4. If config tuning isn't enough, add a **post-association module** (see File
   ownership for where to put it).

## Measurement / success criteria

- Use `src/eval_clip.py` on `1minclip.mov` for before/after every change.
- Success = id_switches and track_id_change_rate down meaningfully (target
  track_id_change_rate < 0.40, the config threshold in `EVAL_MAX_TRACK_ID_CHANGE_RATE`)
  with **no regression** in avg_detections / zero_det_rate.
- Log every before/after number; don't silently accept a change.

## File ownership (do NOT edit — other agents own these)

- `src/post_process.py` — HUD agent owns it (incl. `suppress_duplicate_tracks`;
  if you need dedup/re-assoc changes, put them in a NEW module instead).
- `src/detection.py` — HUD agent owns it. The `model.track()` call lives here but
  you should **not** need to edit it: tracker choice/params come from the yaml
  (`tracker=tracker` arg) and the `--tracker` CLI flag. If you must add
  post-association code, create a **new file** `src/track_reassoc.py` and wire it
  into `src/pipeline.py` (coordinate the one-line pipeline call).
- `src/config.py` `SEG_MODEL_PATH` / `PLAYER_PREDICT_CONF` — training agent owns.
- You own: `configs/bytetrack.yaml`, `configs/botsort.yaml`, a new
  `src/track_reassoc.py` if needed, and the tracker-default selection.

## Coordination warning

All three agents share ONE working tree. If you and the HUD agent both edit
`src/detection.py` or `src/post_process.py` you WILL clobber each other. Keep
your code changes in `configs/*.yaml` and a new module. If you truly must edit a
shared file, tell the user so they can serialize it. Strongly consider asking the
user to put you on a separate `git worktree`.

## Constraints

- One concern per commit. Don't push; commits stay local until all three
  workstreams are validated together.
