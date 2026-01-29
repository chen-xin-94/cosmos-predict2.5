
"""
Preprocess Data Foundry dataset to JSON format for Cosmos Predict2.5 training.
"""

import argparse
import json
import os

import pandas as pd

# TODO: multiprocessing for speedup

# === Configurations ===
CONFIGS = {
    "data_foundry": {
        "camera_views": [
            "observation.images.frame_camera_left",
            "observation.images.frame_camera_top",
            "observation.images.wrist_camera",
        ],
        "state_key": "observation.state.franka_robot_ee",
        "gripper_key": "observation.state.franka_robot_gripper",
        "action_key": "action",
        "append_last_action_to_state": True,
    }
}

# === Base Paths ===
parser = argparse.ArgumentParser(description="Convert parquet episodes to JSON annotations.")
parser.add_argument(
    "--dataset-subdir",
    default="avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand",
    help="Dataset subdirectory under the data_foundry root.",
)
parser.add_argument(
    "--config-name",
    default="data_foundry",
    choices=CONFIGS.keys(),
    help="Name of the configuration to use for dataset keys.",
)
args = parser.parse_args()

# Load selected config
if args.config_name not in CONFIGS:
    raise ValueError(f"Config '{args.config_name}' not found. Available configs: {list(CONFIGS.keys())}")
current_config = CONFIGS[args.config_name]

data_root = os.path.join(
    "/mnt/central_storage/data_pool/data_foundry",
    args.dataset_subdir,
)
root_parquet_dir = os.path.join(data_root, "data")
root_video_dir = os.path.join(data_root, "videos")

output_root = os.path.join(
    "/raid/chen.xin/repo/cosmos-predict2.5/datasets/df",
    args.dataset_subdir,
    "annotation/all",
)
os.makedirs(output_root, exist_ok=True)

meta_path = os.path.join(data_root, "meta/episodes.jsonl")


# ------------------------------------------------------------
# Load metadata → dict: episode_index → {task, text}
# ------------------------------------------------------------
episode_meta = {}

if os.path.exists(meta_path):
    with open(meta_path, "r") as f:
        for line in f:
            d = json.loads(line)

            idx = d["episode_index"]
            task_raw = d["tasks"][0]

            # Simplified: everything in tasks[0] goes to "text" as per user request.
            episode_meta[idx] = {
                "text": task_raw.strip(),
            }
else:
    print(f"[WARNING] Meta file not found at {meta_path}. Text fields will be empty.")


# ------------------------------------------------------------
# Process 1 parquet → JSON (NO DOWNSAMPLING)
# ------------------------------------------------------------
def process_single_parquet(parquet_path, save_path, episode_index, chunk_name):
    df = pd.read_parquet(parquet_path)

    # Keep ALL frames (no downsampling)
    df_filtered = df.reset_index(drop=True)

    # Use configured camera views
    camera_views = current_config["camera_views"]

    # Build videos list with all camera views
    videos_list = []
    for cam_view in camera_views:
        # Construct video path
        # Note: Lerobot typically structures videos as videos/chunk/camera_key/episode_id.mp4
        # We assume this structure holds for all supported datasets.
        video_abs_path = os.path.join(
            root_video_dir,
            chunk_name,
            cam_view,
            f"episode_{episode_index:06d}.mp4",
        )
        if not os.path.exists(video_abs_path):
            print(f"[WARNING] Missing video file: {video_abs_path}")
        videos_list.append({"video_path": video_abs_path})

    # Build JSON dict in your required order
    output = {
        "text": episode_meta.get(episode_index, {"text": ""})["text"],
        "videos": videos_list,
        "state": [],
        "continuous_gripper_state": [],
        "episode_index": episode_index,
    }

    # Get keys from config
    state_key = current_config["state_key"]
    gripper_key = current_config["gripper_key"]
    action_key = current_config.get("action_key", "action")
    append_last_action_to_state = current_config.get("append_last_action_to_state", False)

    # Fill state + gripper lists using ALL rows
    for _, row in df_filtered.iterrows():
        # 7D EE state
        if state_key in row:
            ee = [float(x) for x in row[state_key]]
        else:
             # Fallback or error if key missing? For now assumes it must exist or error out
            raise KeyError(f"State key '{state_key}' not found in parquet columns: {row.index.tolist()}")

        # Add last action dim
        extra_dims = []
        if append_last_action_to_state:
            # For the default Bridge dataset and data foundry dataset, _get_actions actually use state to calculate action instead of loading action from json,
            # so we add last dim of action here to show gripper value which is not present in state values in some dataset e.g. data foundry.
            # However, in _get_robot_states, arm_states = states[:, :6] for bridge dataset, so it actually doesn't matter if we add the last action dim here or not. 
            if action_key in row:
                action_last = float(row[action_key][-1])
                extra_dims.append(action_last)
            else:
                extra_dims.append(0.0) # Placeholder value
                print(f"[WARNING] '{action_key}' column missing in {parquet_path}")

        output["state"].append(ee + extra_dims)

        # Gripper scalar
        if gripper_key in row:
            grip = float(row[gripper_key])
        else:
             raise KeyError(f"Gripper key '{gripper_key}' not found in parquet columns")
        
        output["continuous_gripper_state"].append(grip)

    # Add timesteps = number of rows
    output["timesteps"] = len(output["state"])

    # Save JSON
    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"[OK] Saved {save_path}")


# ------------------------------------------------------------
# Loop over all chunks
# ------------------------------------------------------------
for chunk_name in sorted(os.listdir(root_parquet_dir)):
    chunk_path = os.path.join(root_parquet_dir, chunk_name)
    if not os.path.isdir(chunk_path):
        continue

    for fname in sorted(os.listdir(chunk_path)):
        if not fname.endswith(".parquet"):
            continue

        parquet_path = os.path.join(chunk_path, fname)
        episode_index = int(fname.replace("episode_", "").replace(".parquet", ""))

        save_name = f"{episode_index:06d}.json"
        save_path = os.path.join(output_root, save_name)

        try:
            process_single_parquet(parquet_path, save_path, episode_index, chunk_name)
        except Exception as e:
            print(f"[ERROR] {parquet_path}: {e}")
