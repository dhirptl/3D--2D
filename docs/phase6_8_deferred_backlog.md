# Deferred Phase 6-8 Backlog

This backlog captures TDD phases intentionally deferred until core metrics stabilize.

## Entry Criteria (must be true first)

- Broadcast gate passes on held-out clips:
  - FP rate < 20%
  - Zero-det frame rate < 5%
- Pass 1/Pass 2 pipeline is stable and used by default for offline analysis.
- Route metrics improve vs baseline:
  - lower fragmentation
  - low ID-switch proxy
  - team labeling at or above current baseline

## Phase 6: Jersey OCR

- Add torso/back crop extraction from Pass 1 detections.
- Add OCR inference restricted to jersey digits (0-99).
- Aggregate OCR votes per `stable_id`.
- Integrate OCR constraints into stitch gating (same number favors merge; conflicting confident numbers block merge).
- Acceptance:
  - measurable drop in ID-switch proxy on occlusion-heavy clips.

## Phase 7: Export + batching

- Export seg/pose/helmet models to target runtime (CoreML on Apple, TensorRT on CUDA).
- Batch-frame inference in Pass 1.
- Re-validate quality parity vs eager PyTorch outputs.
- Acceptance:
  - Pass 1 throughput improves with no regression in route metrics.

## Phase 8: Automatic field registration

- Improve landmark matching robustness in `field_registration.py`.
- Use camera-motion composition between expensive re-estimation steps.
- Validate on multi-broadcast holdout clips.
- Acceptance:
  - remove manual homography requirement for most clips.
