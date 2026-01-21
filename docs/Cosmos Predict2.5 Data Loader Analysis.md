# Cosmos Predict2.5 Data Loader Analysis

This document provides comprehensive analysis of the three video generation implementations in cosmos-predict2.5 with full code references.

---

## 1. Implementation Overview

### 1.1 Standard (pre-trained/post-trained)
- **Config path**: `predict2/configs/video2world/`
- **Model**: [Video2WorldModelRectifiedFlow](cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_model.py#65-295)
- **Dataset**: WebDataset via [get_video_dataset()](cosmos_predict2/_src/predict2/datasets/dataset_provider.py#41-167)
- **Inference**: [Video2WorldInference](cosmos_predict2/_src/predict2/inference/video2world.py#236-825) in [video2world.py](cosmos_predict2/_src/predict2/inference/video2world.py)

### 1.2 Action-Conditioned (robot/action-cond)
- **Config path**: `predict2/action/configs/`
- **Model**: [ActionVideo2WorldModelRectifiedFlow](cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_model.py#65-295)
- **Dataset**: [Dataset_3D](cosmos_predict2/_src/predict2/action/datasets/dataset_local.py#46-430) (local JSON + MP4 files)
- **Inference**: [ActionVideo2WorldInference](cosmos_predict2/_src/predict2/action/inference/inference_pipeline.py#31-361) in [inference_pipeline.py](cosmos_predict2/_src/predict2/action/inference/inference_pipeline.py)

### 1.3 Multi-view (auto/multiview)
- **Config path**: `predict2_multiview/configs/`
- **Model**: [MultiviewVid2VidModelRectifiedFlow](cosmos_predict2/_src/predict2_multiview/models/multiview_vid2vid_model_rectified_flow.py#61-332)
- **Dataset**: WebDataset via [get_multiview_video_loader()](cosmos_predict2/_src/predict2_multiview/datasets/multiview.py#471-505)
- **Inference**: [Vid2VidInference](cosmos_predict2/_src/predict2_multiview/scripts/inference.py#98-218) in [inference.py](cosmos_predict2/_src/predict2_multiview/scripts/inference.py)

---

## 2. Frame Counts & Latent Dimensions

### 2.1 Latent Dimension Calculation (All Implementations)

The VAE (Wan2.1) uses temporal compression factor 4 with special formula:

```python
# cosmos_predict2/_src/predict2/tokenizers/wan2pt1.py#L1028-L1032
def get_latent_num_frames(self, num_pixel_frames: int) -> int:
    return 1 + (num_pixel_frames - 1) // 4

def get_pixel_num_frames(self, num_latent_frames: int) -> int:
    return (num_latent_frames - 1) * 4 + 1
```

### 2.2 Standard Implementation

```python
# cosmos_predict2/_src/predict2/datasets/dataset_provider.py#L46
num_video_frames: int = 121  # Default (NOT actually used)

# Actual configs ALL use 93 frames:
# cosmos_predict2/_src/predict2/configs/video2world/experiment/reason_embeddings/stage3_2B.py#L337
num_video_frames=93,  # This is the real value
state_t = 24  # = 1 + (93-1)//4
```

> **Note**: The default `121` in [dataset_provider.py](cosmos_predict2/_src/predict2/datasets/dataset_provider.py) is overridden by experiment configs. All Standard configs use **93 raw frames** (matching the paper).

### 2.3 Action-Conditioned Implementation

```python
# cosmos_predict2/_src/predict2/action/datasets/dataset_local.py#L127
self.sequence_length = 1 + num_action_per_chunk  # e.g., 1 + 12 = 13

# cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow.py#L600
state_t = 1 + 12 // 4  # = 4 latent frames
```

| Mode | Raw Frames | Latent (`state_t`) |
|------|------------|-------------------|
| Training | 13 | 4 |
| Inference | Variable (chunked) | Variable |

### 2.4 Multi-view Implementation

```python
# cosmos_predict2/_src/predict2_multiview/configs/vid2vid/defaults/dataloader.py#L52-L56
num_video_frames = [
    ("29frames", 29),   # state_t = 8
    ("61frames", 61),   # state_t = 16
    ("93frames", 93),   # state_t = 24
]
```


---

## 3. FPS Control & Frame Sampling

### 3.1 Standard: Sophisticated FPS Control

**Two decoder options:**

#### Simple filtering ([chunked_video_decoder](cosmos_predict2/_src/predict2/datasets/decoders/video_decoder.py#124-224)):
```python
# cosmos_predict2/_src/predict2/datasets/decoders/video_decoder.py#L182-L186
if video_fps < min_fps_thres:
    raise ValueError(f"Video fps {video_fps} lower than {min_fps_thres}, skipping")
if video_fps > max_fps_thres:
    raise ValueError(f"Video fps {video_fps} larger than {max_fps_thres}, skipping")
```

#### Dynamic stride ([chunked_video_decoder_w_lower_fps](cosmos_predict2/_src/predict2/datasets/decoders/video_decoder.py#295-397)) — **NOT simple filtering:**
```python
# cosmos_predict2/_src/predict2/datasets/decoders/video_decoder.py#L256-L292
def get_frame_indices_w_lowered_fps(
    n_video_frames, video_fps, min_fps_thres, max_fps_thres, n_target_frames
):
    # Calculate valid strides that keep FPS in [min_fps_thres, max_fps_thres]
    for stride in range(min_stride, max_stride + 1):
        new_fps = video_fps / stride
        if min_fps_thres <= new_fps <= max_fps_thres:
            valid_strides.append(stride)

    # 99% probability to pick LARGER stride (lower FPS)
    if len(valid_strides) >= 2:
        stride_choices = valid_strides[-2:]  # Last two = largest strides
        weights = [0.01, 0.99]  # Strongly prefer larger stride
        selected_stride = np.random.choice(stride_choices, p=weights)
```

**Example**: For a 60fps video with `min_fps=10, max_fps=30`:
- Valid strides: 2 (→30fps), 3 (→20fps), 4 (→15fps), 5 (→12fps), 6 (→10fps)
- 99% chance to pick stride=6 (10fps) or stride=5 (12fps)

### 3.2 Action-Conditioned: Fixed Stride with Overlapping Windows

**No FPS thresholds** — uses `fps_downsample_ratio` with sliding window slicing:

The action-conditioned dataset slices each episode into **overlapping windows** at dataset initialization time, not during iteration:

```python
# cosmos_predict2/_src/predict2/action/datasets/dataset_local.py#L206-L227
def _load_and_process_ann_file(self, ann_file):
    samples = []
    n_frames = len(ann[self._state_key])
    
    # OVERLAP: Each window starts every `start_frame_interval` frames (=1 for training)
    for frame_i in range(0, n_frames, self.start_frame_interval):
        sample = {"ann_file": ann_file, "frame_ids": []}
        curr_frame_i = frame_i
        
        while True:
            if curr_frame_i > (n_frames - 1):
                break
            sample["frame_ids"].append(curr_frame_i)
            if len(sample["frame_ids"]) == self.sequence_length:
                break
            curr_frame_i += self.fps_downsample_ratio  # Fixed stride WITHIN window
        
        # Only add if we got a full sequence
        if len(sample["frame_ids"]) == self.sequence_length:
            samples.append(sample)
    return samples
```

#### Key Parameters

| Parameter | Training Value | Meaning |
|-----------|----------------|---------|
| `num_action_per_chunk` | 12 | `sequence_length = 1 + 12 = 13` frames per window |
| `fps_downsample_ratio` | 6 (typical) | Frames sampled every 6 raw frames **within** each window |
| `start_frame_interval` | 1 (hardcoded) | **Overlap stride** — windows start every 1 frame |

#### Visual Example

For an episode with 100 frames, `fps_downsample_ratio=6`, `sequence_length=13`:
- Window 1: frames `[0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]`
- Window 2: frames `[1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73]`
- Window 3: frames `[2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74]`
- ... up to ~28 valid windows (100 - 72 = 28 possible starting points)

#### What Exactly Happens Per Training Iteration

**Each iteration (one backprop) processes a batch of sliced windows, NOT an average over episodes:**

1. **Pre-processing (at init)**: Each episode → multiple overlapping windows stored as separate samples
2. **Per iteration**: DataLoader samples `batch_size` windows (e.g., 4), potentially from **different episodes**
3. **Backprop**: Computed on the batched loss over all windows — no episode-level aggregation

```
Episode 1 (300 frames) → ~228 sliced windows
Episode 2 (200 frames) → ~128 sliced windows
...
Total samples = sum of all windows across all episodes

Training iteration:
  DataLoader → randomly samples batch_size=4 windows (can be from 4 different episodes)
  → Forward pass on batch
  → Single backward pass on batched loss
```

### 3.3 Multi-view: Fixed Stride per View

```python
# cosmos_predict2/_src/predict2_multiview/datasets/multiview.py#L218-L220
frame_end = frame_start + self.num_frames * self.fps_downsample_factor
frame_indices = list(range(frame_start, frame_end, self.fps_downsample_factor))
```

---

## 4. Pre-Clipping & Frame Sampling Strategy (Training)

### 4.1 Standard: Chunk-Based Sampling with Caption Windows

Videos are **pre-annotated with caption chunks** (videos are stored as regular mp4 files but with chunked captions, ~256 frames each). Sampling happens in two stages:

```python
# cosmos_predict2/_src/predict2/datasets/decoders/video_decoder.py#L59-L101
def sample_chunk_index_from_chunked_video(n_video_frames, n_target_frames, chunk_size):
    """
    Videos are stored as regular mp4 files but with chunked captions.
    There is one caption per [chunk_size] frames.
    If the last chunk has frames >= chunk_size / 2, it is treated as separate chunk.
    """
    n_chunks = max(n_video_frames // chunk_size, 1)
    
    # Last chunk handling
    n_frames_in_last_chunk = n_video_frames - n_chunks * chunk_size
    if n_frames_in_last_chunk >= int(0.5 * chunk_size):
        if n_frames_in_last_chunk > n_target_frames:
            n_chunks += 1  # Treat as separate chunk
    
    # Stage 1: Random chunk selection
    sampled_chunk_index = randint(0, n_chunks - 1)
    
    # Get frames available in this chunk
    if sampled_chunk_index == n_chunks - 1:
        n_frames_in_chunk = n_video_frames - sampled_chunk_index * chunk_size
    else:
        n_frames_in_chunk = chunk_size
    
    return sampled_chunk_index, n_frames_in_chunk, "success"
```

```python
# Stage 2: Random start within chunk
# cosmos_predict2/_src/predict2/datasets/decoders/video_decoder.py#L199-L202
chunk_frame_start = sampled_chunk_index * chunk_size
frame_start = chunk_frame_start + int(np.random.choice(n_frames_in_chunk - n_target_frames, 1))
frame_end = frame_start + n_target_frames
```

**What happens to the rest of the video?**
- **Not dropped permanently** — the video exists in the dataset
- **Different chunks used in different epochs** — random chunk selection per [__getitem__](cosmos_predict2/_src/predict2/datasets/local_datasets/dataset_video.py#190-227)
- Caption is associated with the selected chunk

---

### 4.2 Action-Conditioned: Sliding Window (All Possible Segments)

**Pre-processing extracts ALL valid segments** with configurable overlap:

```python
# cosmos_predict2/_src/predict2/action/datasets/dataset_local.py#L206-L227
def _load_and_process_ann_file(self, ann_file):
    samples = []
    n_frames = len(ann[self._state_key])
    
    # Sliding window with start_frame_interval stride
    for frame_i in range(0, n_frames, self.start_frame_interval):
        sample = dict()
        sample["ann_file"] = ann_file
        sample["frame_ids"] = []
        curr_frame_i = frame_i
        
        while True:
            if curr_frame_i > (n_frames - 1):
                break
            sample["frame_ids"].append(curr_frame_i)
            if len(sample["frame_ids"]) == self.sequence_length:
                break
            curr_frame_i += self.fps_downsample_ratio
        
        # Only add if we got full sequence
        if len(sample["frame_ids"]) == self.sequence_length:
            samples.append(sample)
    
    return samples
```

**What happens to the rest of the video?**
- **Nothing is dropped** — all valid 13-frame windows are extracted as separate samples
- Training: `start_frame_interval = 1` (hardcoded in `dataset_local.py#L116`)
- Validation/Test: `val_start_frame_interval = 1` (all configs in `data.py` use 1)
- **Every possible starting frame generates a sample** for both train and val
- For a 100-frame video with 13-frame windows: creates **88 training samples** (100 - 13 + 1)

---

### 4.3 Multi-view: Caption-Based Chunk Selection (`t2w_windows`)

Similar to Standard but uses pre-annotated `t2w_windows`:

```python
# cosmos_predict2/_src/predict2_multiview/datasets/multiview.py#L196-L220
t2w_windows = data[meta_key]["t2w_windows"]

# Random chunk selection (same chunk for all views)
if chunk_index is None:
    chunk_index = random.choice(list(range(len(t2w_windows))))
window = t2w_windows[chunk_index]

# Extract frames from this window
frame_start = window["start_frame"]
frame_end = frame_start + self.num_frames * self.fps_downsample_factor
frame_indices = list(range(frame_start, frame_end, self.fps_downsample_factor))
```

**What happens to the rest of the video?**
- **Not dropped** — different `t2w_windows` used in different epochs
- All camera views use the **same chunk** for temporal consistency
- Each window has associated caption for that time segment

---

### 4.4. Summary: Pre-Clipping Strategies

| Implementation | Pre-Clipping | Sampling Method | Rest of Video |
|----------------|--------------|-----------------|---------------|
| **Standard** | Yes (chunk_size ~256 frames) | Random chunk → random start within chunk | Not dropped, different chunks used per epoch |
| **Action-Cond** | No — sliding window | All valid windows extracted as samples | Fully utilized via sliding window |
| **Multi-view** | Yes (t2w_windows) | Random window selection | Not dropped, different windows per epoch |

---

## 5. Long-horizon Video Generation

### 5.1 Standard: Autoregressive Sliding Window

```python
# cosmos_predict2/_src/predict2/inference/video2world.py#L586-L606
def generate_autoregressive_from_batch(
    self,
    prompt: str,
    input_path: str | torch.Tensor | None,
    num_output_frames: int,
    chunk_size: int,       # Frames per chunk
    chunk_overlap: int,    # Overlap for continuity
    ...
):
    """Generate video using autoregressive sliding window approach."""
```

### 5.2 Action-Conditioned: Segment-Level Autoregressive

```python
# cosmos_predict2/_src/predict2/action/inference/inference.py#L309-L321
for i in range(args.start_frame_idx, len(actions), args.chunk_size):
    next_img_array, video_clamped = video2world_cli.step_inference(
        img_array=img_array,
        action=actions[i : i + args.chunk_size],  # 12 actions per chunk
        guidance=args.guidance,
        seed=i,
    )
    frames.append(next_img_array)
    img_array = next_img_array  # ← Last generated frame → next input
    chunk_video.append(video_clamped)

# Chunk concatenation (skip overlap)
# cosmos_predict2/_src/predict2/action/inference/inference.py#L323-L324
chunk_list = [chunk_video[0]] + [chunk_video[i][:args.chunk_size] for i in range(1, len(chunk_video))]
chunk_video = np.concatenate(chunk_list, axis=0)
```

**Mechanism**:
1. Each [step_inference()](cosmos_predict2/_src/predict2/action/inference/inference_pipeline.py#209-270) generates `chunk_size + 1` frames (e.g., 13)
2. Last generated frame conditions next segment
3. Chunks stitched, taking `chunk_size` frames from each subsequent chunk

### 5.3 Multi-view: Single-Shot Only (No Long-Horizon Mechanism)

```python
# cosmos_predict2/_src/predict2_multiview/scripts/inference.py#L185-L194
sample = self.model.generate_samples_from_batch(
    data_batch,
    state_shape=x0.shape[1:],  # All views at once
    ...
)
# video shape: (B, C, V*T, H, W) — all views concatenated
```

**Current Limitation**: Multi-view has **no autoregressive or chunk-by-chunk mechanism** for long-horizon generation:
- No [autoregressive](cosmos_predict2/_src/predict2/inference/video2world.py#586-815), `sliding`, [overlap](cosmos_predict2/_src/predict2/tokenizers/interface.py#92-95), or [chunk](cosmos_predict2/_src/predict2/tokenizers/interface.py#92-95) logic in the codebase
- Frame count is fixed by training config (29/61/93 per view)
- For longer videos, would need to either:
  1. Train with longer frame configs (93 frames × 7 views = 651 total already memory-intensive)
  2. Implement custom autoregressive wrapper (not currently available)


### 5.4 Summary - Long-horizon Video Generation

| Implementation | Long-Horizon Method | Max Frames |
|---|---|---|
| **Standard** | ✅ `generate_autoregressive_from_batch()` with `chunk_size` and `chunk_overlap` | Unlimited |
| **Action-Cond** | ✅ Segment loop with `chunk_size`, conditioned on last frame | Unlimited (depends on action length) |
| **Multi-view** | ❌ None — single-shot only | 29/61/93 per view (config-fixed) |

---

## 6. Data Pipeline Architecture

### 6.1 Standard Pipeline
```
WebDataset Shards (S3/GCS)
    ↓
video_decoder (with chunk_size for caption alignment)
    - Sample random chunk
    - Sample random start within chunk
    ↓
Augmentor (video_basic_augmentor_v2)
    - FPS adjustment (stride selection)
    - Resize, normalize
    - Caption from chunk
    ↓
DataLoader → Model
```

### 6.2 Action-Conditioned Pipeline
```
Local JSON annotations + MP4 files
    ↓
Dataset_3D.__init__() — pre-extract ALL valid windows
    - Sliding window with start_frame_interval
    - Each window = one sample
    ↓
Dataset_3D.__getitem__()
    - Load video frames for window
    - Extract robot states
    - Compute relative actions
    ↓
DataLoader → Model (with action conditioning)
```

### 6.3 Multi-view Pipeline
```
WebDataset Shards with multi-camera videos
    ↓
ExtractFramesAndCaptions augmentor
    - Random t2w_window selection (same for all views)
    - Per-camera frame extraction
    - Fixed stride (fps_downsample_factor)
    ↓
collate_fn (stack views along temporal dimension)
    ↓
DataLoader → Model (with view indices)
```

---

## 7. Summary Comparison Table

| Aspect | Standard | Action-Conditioned | Multi-view |
|--------|----------|-------------------|------------|
| **Data Source** | WebDataset (S3/GCS) | Local JSON + MP4 | WebDataset |
| **Dataset Class** | [get_video_dataset()](cosmos_predict2/_src/predict2/datasets/dataset_provider.py#41-167) | [Dataset_3D](cosmos_predict2/_src/predict2/action/datasets/dataset_local.py#46-430) | [get_multiview_video_loader()](cosmos_predict2/_src/predict2_multiview/datasets/multiview.py#471-505) |
| **Raw Frames** | **93** (paper value) | 13 (train) | 29/61/93 per view |
| **Latent Frames** | 24 | 4 (train) | 8/16/24 per view |
| **Pre-Clipping** | Yes (chunk_size ~256) | No (sliding window) | Yes (t2w_windows) |
| **Sampling** | Random chunk → random start | All windows as samples | Random window |
| **Rest of Video** | Different chunks per epoch | Fully utilized | Different windows per epoch |
| **FPS Control** | Sophisticated stride selection | Fixed stride | Fixed stride |
| **`min/max_fps_thres`** | ✅ Used (more than filtering) | ❌ Not used | ❌ Not used |
| **`fps_downsample_ratio/factor`** | ❌ | ✅ Used | ✅ Used |
| **Long Video (Training)** | Random temporal crop | Sliding window samples | Per-view windows |
| **Long Video (Inference)** | Autoregressive | Segment-level autoregressive | ❌ None (single-shot only) |
| **Extra Conditioning** | Text | Text + Action (7-DoF) | Text + View indices |
| **Resolution** | Variable | 256×320 / 480×640 | 480p/720p/1080p |

---

## 8. Key File References

### Standard
- **Dataset**: [dataset_provider.py](cosmos_predict2/_src/predict2/datasets/dataset_provider.py)
- **Decoder**: [video_decoder.py](cosmos_predict2/_src/predict2/datasets/decoders/video_decoder.py)
- **Model**: [video2world_model_rectified_flow.py](cosmos_predict2/_src/predict2/models/video2world_model_rectified_flow.py)
- **Inference**: [video2world.py](cosmos_predict2/_src/predict2/inference/video2world.py)
- **Config Example**: [stage3_2B.py](cosmos_predict2/_src/predict2/configs/video2world/experiment/reason_embeddings/stage3_2B.py)

### Action-Conditioned
- **Dataset**: [dataset_local.py](cosmos_predict2/_src/predict2/action/datasets/dataset_local.py)
- **Config**: [exp_2B_action_conditioned_rectify_flow.py](cosmos_predict2/_src/predict2/action/configs/action_conditioned/experiment/exp_2B_action_conditioned_rectify_flow.py)
- **Model**: [action_conditioned_video2world_rectified_flow_model.py](cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_rectified_flow_model.py)
- **Inference**: [inference.py](cosmos_predict2/_src/predict2/action/inference/inference.py)
- **Pipeline**: [inference_pipeline.py](cosmos_predict2/_src/predict2/action/inference/inference_pipeline.py)

### Multi-view
- **Dataset**: [multiview.py](cosmos_predict2/_src/predict2_multiview/datasets/multiview.py)
- **Config**: [buttercup2p5_rectified_flow.py](cosmos_predict2/_src/predict2_multiview/configs/vid2vid/experiment/buttercup2p5_rectified_flow.py)
- **Model**: [multiview_vid2vid_model_rectified_flow.py](cosmos_predict2/_src/predict2_multiview/models/multiview_vid2vid_model_rectified_flow.py)
- **Inference**: [inference.py](cosmos_predict2/_src/predict2_multiview/scripts/inference.py)

### Tokenizer (Shared)
- **VAE**: [wan2pt1.py](cosmos_predict2/_src/predict2/tokenizers/wan2pt1.py)
