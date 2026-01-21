import os
import pandas as pd
import json

# === Base Paths ===
data_root = "/mnt/central_storage/data_pool/data_foundry/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand"
root_parquet_dir = os.path.join(data_root, "data")
root_video_dir = os.path.join(data_root, "videos")

output_root = "/raid/yusong/workspace/cosmos-predict2.5_df/cosmos_predict2/datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand/annotation/processed_json"
os.makedirs(output_root, exist_ok=True)

meta_path = os.path.join(data_root, "meta/episodes_original.jsonl")


# ------------------------------------------------------------
# Load metadata → dict: episode_index → {task, text}
# ------------------------------------------------------------
episode_meta = {}

with open(meta_path, "r") as f:
    for line in f:
        d = json.loads(line)

        idx = d["episode_index"]
        task_raw = d["tasks"][0]

        if ":" in task_raw:
            task_root, text = task_raw.split(":", 1)
            task_root = task_root.strip()
            text = text.strip()
        else:
            task_root = task_raw.strip()
            text = ""

        episode_meta[idx] = {
            "task": task_root,
            "text": text,
        }


# ------------------------------------------------------------
# Process 1 parquet → JSON (NO DOWNSAMPLING)
# ------------------------------------------------------------
def process_single_parquet(parquet_path, save_path, episode_index, chunk_name):
    df = pd.read_parquet(parquet_path)

    # Keep ALL frames (no downsampling)
    df_filtered = df.reset_index(drop=True)

    # Define camera view folder names
    camera_views = [
        "observation.images.frame_camera_left",
        "observation.images.frame_camera_top",
        "observation.images.wrist_camera",
    ]

    # Build videos list with all camera views
    videos_list = []
    for cam_view in camera_views:
        video_abs_path = os.path.join(
            root_video_dir,
            chunk_name,
            cam_view,
            f"episode_{episode_index:06d}.mp4",
        )
        if not os.path.exists(video_abs_path):
            print(f"[WARNING] Missing video file: {video_abs_path}")
        videos_list.append({"video_path": video_abs_path})

    # Episode metadata
    meta = episode_meta.get(episode_index, {"task": "", "text": ""})

    # Build JSON dict in your required order
    output = {
        "task": meta["task"],
        "text": meta["text"],
        "videos": videos_list,
        "state": [],
        "continuous_gripper_state": [],
        "episode_index": episode_index,
    }

    # Fill state + gripper lists using ALL rows
    for _, row in df_filtered.iterrows():
        # 7D EE state
        ee = [float(x) for x in row["observation.state.franka_robot_ee"]]

        # Add last action dim
        action_last = float(row["action"][-1])
        output["state"].append(ee + [action_last])

        # Gripper scalar
        grip = float(row["observation.state.franka_robot_gripper"])
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
