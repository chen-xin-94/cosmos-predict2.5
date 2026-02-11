# Add a New LeRobot Dataset (Action-Conditioned, Multi-View)

This guide summarizes **all code changes** needed to add a new LeRobot dataset to the current action-conditioned training pipeline, based on the full DROID integration workflow.

## Scope

Use this when the source dataset is LeRobot v2.x style (Parquet + videos + `meta/`).

Expected outcomes:
- Preprocessed JSON annotations for training.
- Dataset class that loads direct actions (no complex action reconstruction logic).
- Registered Hydra dataloaders.
- Full + smoke experiments.
- Smoke subset under `datasets/smoke_test/<dataset_name>`.

---

## 1) Preprocessing Changes

Primary file:
- `scripts/preprocessing/preprocess_lerobot.py`

What to add/update:
1. Add a config entry in `LEROBOT_CONFIGS`:
- `dataset_path`
- `output_path`
- `stats_path`
- `stats_filename`
- `state_key` (example: `observation.state`)
- `action_key` (example: `action`)
- `camera_views` (ordered list of 3 view keys)

2. Ensure text mapping is correct:
- Read `task_index` from episode parquet.
- Map it to text via `meta/tasks.jsonl`.

3. Ensure chunk/video path mapping is correct:
- Data source: `data/chunk-xxx/episode_xxxxxx.parquet`.
- Video source: `videos/chunk-xxx/<camera_view>/episode_xxxxxx.mp4`.

4. Keep multiprocessing support:
- Use process workers for speed.
- Keep fallback to threads if process spawning is blocked.

5. Generate aggregated stats with timestep-weighted aggregation:
- Use per-episode summary (`n`, `mean`, `var`, `min`, `max`).
- Aggregate with weighted sums across episodes.
- Save to `assets/action_conditioned/concat_view/<dataset_name>/stats.json`.

6. Keep metadata traceability:
- Read `meta/stats.json` and include reference block in output stats.

Notes:
- This script currently writes `annotation/all` first, then split is done separately.
- For a new dataset, validate one JSON file manually before mass processing.

---

## 2) Splitting Annotations

Split script:
- `scripts/preprocessing/split.py`

Recommended structure for training code:
- `datasets/<dataset_name>/annotation/train`
- `datasets/<dataset_name>/annotation/val`
- `datasets/<dataset_name>/annotation/test`

If you only have train/test:
- Keep `val` empty or missing.
- Add fallback logic in dataloader config (see section 4) so validation uses `test`.

---

## 3) New Dataset Class (Loader)

Create file:
- `cosmos_predict2/_src/predict2/action/datasets/dataset_<dataset_name>.py`

Pattern used for LeRobot direct-action datasets:
1. Inherit from `Dataset_3D_DF` for shared video loading path behavior.
2. Override item loading to:
- Load `action` directly from JSON.
- Select `frame_ids[1:]` actions for a chunk.
- Normalize action using stats file (usually min-max).
3. Keep text loading simple (`label.get("text", "")`).
4. Keep output dict interface compatible with existing training pipeline.
5. Add multiview subclass that concatenates views in image space along width.

DROID reference:
- `cosmos_predict2/_src/predict2/action/datasets/dataset_droid.py`

---

## 4) Register Data in Hydra

Main file:
- `cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py`

Required additions:
1. Import your dataset class.
2. Add dataset path constants:
- full train/val/test paths
- smoke train/val/test paths
- action stats path

3. Add `val` fallback helper:
- If `annotation/val` has no JSON files, fallback to `annotation/test`.

4. Add per-view resolution loading from LeRobot metadata:
- Read `/mnt/.../<dataset>/meta/info.json`.
- Use `features.<camera_key>.shape[:2]` as `[H, W]`.
- Keep a safe default fallback.
- Align `[H, W]` to tokenizer spatial compression factor (`16`) before dataset registration.
  - Recommended policy: floor to nearest multiple of 16 (for both height and width).
  - Emit a warning when alignment changes the metadata resolution.

5. Register full dataloaders with stable names, e.g.:
- `<dataset>_multiview_13frame_<H>_<3W>_train`
- `<dataset>_multiview_13frame_<H>_<3W>_val`

6. Register smoke dataloaders with separate names, e.g.:
- `<dataset>_multiview_13frame_<H>_<3W>_smoke_train`
- `<dataset>_multiview_13frame_<H>_<3W>_smoke_val`

DROID reference blocks:
- Full registration in `data.py` (`droid_multiview_13frame_176_960_*`)
- Smoke registration in `data.py` (`droid_multiview_13frame_176_960_smoke_*`)

---

## 5) Add Full + Smoke Experiments

Main file:
- `cosmos_predict2/experiments/base/action.py`

Add a full experiment:
- Name: `ac_reason_embeddings_rectified_flow_2b_<dataset>_<H>_<3W>`
- Override:
  - `/data_train`
  - `/data_val`
- Set model action dims correctly (for LeRobot droid-style, `action_dim=7`).
- Set dataloader override video size to per-view `[H, W]`.

Add a smoke experiment:
- Name: `ac_reason_embeddings_rectified_flow_2b_<dataset>_<H>_<3W>_smoke`
- Inherit from full experiment.
- Override `data_train/data_val` to smoke dataloaders.
- Use short `max_iter`, frequent logging/sampling, small save interval.

Finally register both in `experiments = { ... }` map.

DROID references:
- Full: `ac_reason_embeddings_rectified_flow_2b_droid_176_960`
- Smoke: `ac_reason_embeddings_rectified_flow_2b_droid_176_960_smoke`

---

## 6) Training Scripts

Add scripts under `scripts/train/`:
- `train_ac_mv_<dataset>.sh`
- `train_ac_mv_<dataset>_smoke.sh`

Must match experiment names exactly.

DROID references:
- `scripts/train/train_ac_mv_droid.sh`
- `scripts/train/train_ac_mv_droid_smoke.sh`

---

## 7) Smoke Dataset Construction

Create local smoke subset:
- `datasets/smoke_test/<dataset_name>/annotation/train`
- `datasets/smoke_test/<dataset_name>/annotation/val`
- `datasets/smoke_test/<dataset_name>/annotation/test`

Recommended:
- Copy 10 JSON files per split.
- If source `val` is empty, copy from source `test` for smoke `val`.
- Verify each JSON references existing video files.

DROID implementation used:
- 10 train + 10 val + 10 test JSON files under `datasets/smoke_test/droid/annotation/`.

---

## 8) Resolution Policy (Important)

Do not assume resolution from old experiments.

For LeRobot datasets, use:
- `/mnt/.../<dataset>/meta/info.json`
- `features.<camera_key>.shape = [H, W, C]`

Then enforce model-compatible size:
- Validate `H % 16 == 0` and `W % 16 == 0`.
- If not divisible, align to `[H - H % 16, W - W % 16]` in data config.
- Use the aligned resolution consistently in:
  - dataset registration names (`<H>_<3W>`)
  - experiment names
  - train/smoke shell scripts
  - dataloader override `video_size`

DROID example:
- Metadata per-view: `180x320`
- Effective per-view (16-aligned): `176x320`
- 3-view horizontal concat (effective): `176x960`

---

## 9) Validation Checklist

Run at minimum:
1. Python compile check for edited files.
2. Smoke annotation integrity check:
- 10 files per split.
- All `videos[*].video_path` exist.
- `state` and `action` non-empty.
3. Verify experiment and dataloader names are consistent across:
- `data.py`
- `experiments/base/action.py`
- `scripts/train/*.sh`
4. Verify resolution divisibility and consistency:
- Per-view `H/W` are multiples of 16.
- Concat width equals `num_views * per_view_width`.

If runtime environment lacks dependencies (e.g., `torch`, `attrs`, CUDA extras), compile + static checks are still required, and runtime checks should be deferred to the training environment.

---

## 10) Common Pitfalls

1. Empty `annotation/val` causes val dataloader issues unless fallback is implemented.
2. Experiment name changed but train shell script still points to old name.
3. Dataloader registration name mismatch with experiment override.
4. Using raw `info.json` resolution directly when it is not divisible by 16 (causes implicit truncation/cropping downstream).
5. Smoke experiment still points to full dataset instead of smoke dataset.
6. Action normalization stats path missing or stale.

---

## 11) Optional: Inference Integration

If you also need action-conditioned inference for the new dataset, update:
- Inference params file under `assets/action_conditioned/concat_view/<dataset_name>/inference_params.json`.
- `action_load_fn` in `cosmos_predict2/action_conditioned.py` if loading behavior differs.

For direct-action LeRobot datasets (like DROID-style), ensure:
- Resolution matches training (`H,3W`).
- Action scaling/normalization assumptions match training pipeline.
- Multi-view frame concatenation order matches `camera_views` order used in preprocessing.

---

## Quick File Map

Core files touched when adding a new LeRobot dataset:
- `scripts/preprocessing/preprocess_lerobot.py`
- `scripts/preprocessing/split.py`
- `cosmos_predict2/_src/predict2/action/datasets/dataset_<dataset_name>.py`
- `cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py`
- `cosmos_predict2/experiments/base/action.py`
- `scripts/train/train_ac_mv_<dataset_name>.sh`
- `scripts/train/train_ac_mv_<dataset_name>_smoke.sh`
- `datasets/smoke_test/<dataset_name>/annotation/*`
