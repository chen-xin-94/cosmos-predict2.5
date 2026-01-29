# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Preprocess AgiBotWorld dataset to JSON format for Cosmos Predict2.5 training.

Features:
- Two-pass processing: first compute task-wise statistics, then normalize
- Proper aggregation: weighted mean, combined variance, global min/max
- Saves normalized state/action in [0, 1] range

Usage:
    python scripts/preprocessing/preprocess_agibot.py \
        --task-ids 352 \
        --src-path /mnt/central_storage/data_pool_raw/AgiBotWorld-Alpha \
        --output-path datasets/agibot/annotation \
        --stats-path assets/action_conditioned/concat_view/agibot/stats \
        --debug
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from tqdm import tqdm

from agibot_config import ACTION_KEYS, STATE_KEYS, VIDEO_VIEWS


def load_task_info(task_json_path: Path) -> dict[int, dict]:
    """Load task info JSON and return dict keyed by episode_id."""
    with open(task_json_path, "r") as f:
        task_info: list = json.load(f)
    return {episode["episode_id"]: episode for episode in task_info}


def load_h5_data(h5_path: Path, keys: list[str], prefix: str) -> np.ndarray:
    """
    Load and flatten data from H5 file for given keys.
    
    Args:
        h5_path: Path to the H5 file
        keys: List of keys like "effector.position", "end.orientation"
        prefix: "state" or "action"
    
    Returns:
        np.ndarray of shape (T, flattened_dim)
    """
    with h5py.File(h5_path, "r") as f:
        arrays = []
        num_frames = None
        
        for key in keys:
            h5_key = f"{prefix}/{key.replace('.', '/')}"
            
            if h5_key not in f:
                print(f"[WARNING] Key {h5_key} not found in {h5_path}")
                continue
                
            data = np.array(f[h5_key], dtype=np.float32)
            
            if data.size == 0:
                continue
            
            if num_frames is None:
                num_frames = data.shape[0]
            elif data.shape[0] != num_frames:
                print(f"[WARNING] Mismatched frames for {h5_key}: {data.shape[0]} vs {num_frames}")
                continue
            
            if len(data.shape) > 1:
                data = data.reshape(num_frames, -1)
            else:
                data = data.reshape(num_frames, 1)
            
            arrays.append(data)
        
        if not arrays:
            return np.array([])
        
        return np.concatenate(arrays, axis=1)


def get_video_paths(obs_dir: Path) -> list[dict[str, str]]:
    """Get video paths for an episode in the specified order."""
    video_dir = obs_dir / "videos"
    videos = []
    
    for view in VIDEO_VIEWS:
        video_path = video_dir / f"{view}.mp4"
        videos.append({"video_path": str(video_path)})
    
    return videos


def compute_episode_stats(data: np.ndarray) -> dict:
    """Compute per-episode statistics for aggregation."""
    return {
        "n": data.shape[0],  # frame count
        "mean": np.mean(data, axis=0),
        "var": np.var(data, axis=0),
        "min": np.min(data, axis=0),
        "max": np.max(data, axis=0),
    }


def aggregate_task_stats(episode_stats_list: list[dict]) -> dict:
    """
    Aggregate episode statistics to compute task-wide statistics.
    
    Aggregation Method:
    - Mean: Weighted average of episode means (weighted by frame count)
    - Std: Combined variance accounting for intra-episode variance and variance between episode means
    - Min/Max: Global min/max across all episodes
    """
    if not episode_stats_list:
        return {}
    
    # Total frame count
    total_n = sum(s["n"] for s in episode_stats_list)
    
    # Weighted mean: sum(n_i * mean_i) / total_n
    weighted_mean = np.zeros_like(episode_stats_list[0]["mean"])
    for s in episode_stats_list:
        weighted_mean += s["n"] * s["mean"]
    weighted_mean /= total_n
    
    # Combined variance using parallel algorithm
    # Var = E[Var_i] + Var[Mean_i]
    # where E[Var_i] = weighted average of within-episode variances
    # and Var[Mean_i] = weighted variance of episode means
    
    # Within-episode variance (weighted average)
    within_var = np.zeros_like(weighted_mean)
    for s in episode_stats_list:
        within_var += s["n"] * s["var"]
    within_var /= total_n
    
    # Between-episode variance (variance of means, weighted)
    between_var = np.zeros_like(weighted_mean)
    for s in episode_stats_list:
        diff = s["mean"] - weighted_mean
        between_var += s["n"] * (diff ** 2)
    between_var /= total_n
    
    # Combined variance and std
    combined_var = within_var + between_var
    combined_std = np.sqrt(combined_var)
    
    # Global min/max
    global_min = episode_stats_list[0]["min"].copy()
    global_max = episode_stats_list[0]["max"].copy()
    for s in episode_stats_list[1:]:
        global_min = np.minimum(global_min, s["min"])
        global_max = np.maximum(global_max, s["max"])
    
    return {
        "mean": weighted_mean.tolist(),
        "std": combined_std.tolist(),
        "min": global_min.tolist(),
        "max": global_max.tolist(),
        "total_frames": int(total_n),
        "num_episodes": len(episode_stats_list),
    }


def normalize_minmax(data: np.ndarray, stats: dict) -> np.ndarray:
    """Normalize data to [0, 1] range using min-max scaling."""
    min_vals = np.array(stats["min"])
    max_vals = np.array(stats["max"])
    
    # Avoid division by zero
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals == 0, 1.0, range_vals)
    
    normalized = (data - min_vals) / range_vals
    return np.clip(normalized, 0.0, 1.0)


def collect_task_statistics(
    task_id: int,
    src_path: Path,
    debug: bool = False,
) -> tuple[dict, dict, list[tuple[int, dict]]]:
    """
    First pass: collect statistics from all episodes in a task.
    
    Returns:
        - state_stats: aggregated state statistics
        - action_stats: aggregated action statistics
        - episode_data: list of (episode_id, episode_info) for second pass
    """
    task_json_path = src_path / "task_info" / f"task_{task_id}.json"
    
    if not task_json_path.exists():
        print(f"[ERROR] Task info not found: {task_json_path}")
        return {}, {}, []
    
    task_info = load_task_info(task_json_path)
    episode_ids = sorted(task_info.keys())
    
    if debug:
        episode_ids = episode_ids[:1]
    
    print(f"\n[Pass 1] Collecting statistics for task {task_id}: {len(episode_ids)} episodes")
    
    state_stats_list = []
    action_stats_list = []
    valid_episodes = []
    
    for episode_id in tqdm(episode_ids, desc=f"Task {task_id} stats"):
        obs_dir = src_path / "observations" / str(task_id) / str(episode_id)
        proprio_path = src_path / "proprio_stats" / str(task_id) / str(episode_id) / "proprio_stats.h5"
        
        if not obs_dir.exists() or not proprio_path.exists():
            continue
        
        # Check videos exist
        video_paths = get_video_paths(obs_dir)
        if not all(Path(vp["video_path"]).exists() for vp in video_paths):
            continue
        
        # Load data
        state_data = load_h5_data(proprio_path, STATE_KEYS, "state")
        action_data = load_h5_data(proprio_path, ACTION_KEYS, "action")
        
        if state_data.size == 0 or action_data.size == 0:
            continue
        
        # Validate dimensions (skip episodes with inconsistent dims)
        expected_state_dim = 55  # From config
        expected_action_dim = 36 # From config
        if state_data.shape[1] != expected_state_dim:
            print(f"[WARNING] Episode {episode_id}: state dim {state_data.shape[1]} != {expected_state_dim}, skipping")
            continue
        if action_data.shape[1] != expected_action_dim:
            print(f"[WARNING] Episode {episode_id}: action dim {action_data.shape[1]} != {expected_action_dim}, skipping")
            continue
        
        # Collect stats
        state_stats_list.append(compute_episode_stats(state_data))
        action_stats_list.append(compute_episode_stats(action_data))
        valid_episodes.append((episode_id, task_info.get(episode_id, {})))
    
    # Aggregate statistics
    state_stats = aggregate_task_stats(state_stats_list)
    action_stats = aggregate_task_stats(action_stats_list)
    
    return state_stats, action_stats, valid_episodes


def process_and_save_episodes(
    task_id: int,
    src_path: Path,
    output_path: Path,
    state_stats: dict,
    action_stats: dict,
    valid_episodes: list[tuple[int, dict]],
    debug: bool = False,
) -> int:
    """
    Second pass: normalize and save episodes using computed statistics.
    """
    print(f"\n[Pass 2] Processing and saving task {task_id}: {len(valid_episodes)} episodes")
    
    processed_count = 0
    
    for episode_id, episode_info in tqdm(valid_episodes, desc=f"Task {task_id} save"):
        obs_dir = src_path / "observations" / str(task_id) / str(episode_id)
        proprio_path = src_path / "proprio_stats" / str(task_id) / str(episode_id) / "proprio_stats.h5"
        
        # Load data
        state_data = load_h5_data(proprio_path, STATE_KEYS, "state")
        action_data = load_h5_data(proprio_path, ACTION_KEYS, "action")
        
        # Normalize to [0, 1]
        state_normalized = normalize_minmax(state_data, state_stats)
        action_normalized = normalize_minmax(action_data, action_stats)
        
        # Get video paths
        video_paths = get_video_paths(obs_dir)
        
        # Get timesteps
        with h5py.File(proprio_path, "r") as f:
            timesteps = len(f["timestamp"])
        
        # Build output
        output = {
            "text": episode_info.get("task_name", ""),
            "videos": video_paths,
            "state": state_normalized.tolist(),
            "action": action_normalized.tolist(),
            "episode_index": episode_id,
            "timesteps": timesteps,
            "label_info": episode_info.get("label_info", {}),
        }
        
        # Save to JSON file
        task_output_dir = output_path / f"task_{task_id}" / "all"
        task_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = task_output_dir / f"{episode_id}.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=4)
        
        processed_count += 1
        
        if debug:
            print(f"\n[DEBUG] Processed episode {episode_id}")
            print(f"  - State range: [{state_normalized.min():.4f}, {state_normalized.max():.4f}]")
            print(f"  - Action range: [{action_normalized.min():.4f}, {action_normalized.max():.4f}]")
    
    return processed_count


def save_stats(stats_path: Path, task_id: int, state_stats: dict, action_stats: dict):
    """Save aggregated statistics to JSON file."""
    stats_path.mkdir(parents=True, exist_ok=True)
    
    stats = {
        "task_id": task_id,
        "state": state_stats,
        "action": action_stats,
    }
    
    output_file = stats_path / f"task_{task_id}.json"
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"[OK] Saved stats to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess AgiBotWorld dataset to JSON format")
    parser.add_argument(
        "--task-ids",
        type=int,
        nargs="+",
        required=True,
        help="List of task IDs to process (e.g., 352 353)",
    )
    parser.add_argument(
        "--src-path",
        type=Path,
        default=Path("/mnt/central_storage/data_pool_raw/AgiBotWorld-Alpha"),
        help="Path to raw AgiBotWorld dataset",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output directory for JSON files",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        default=Path("assets/action_conditioned/concat_view/agibot"),
        help="Output directory for aggregated statistics",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: process only first episode per task",
    )
    args = parser.parse_args()
    
    # Create output directories
    args.output_path.mkdir(parents=True, exist_ok=True)
    args.stats_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Source path: {args.src_path}")
    print(f"Output path: {args.output_path}")
    print(f"Stats path: {args.stats_path}")
    print(f"Task IDs: {args.task_ids}")
    print(f"Debug mode: {args.debug}")
    print("-" * 50)
    
    total_processed = 0
    
    for task_id in args.task_ids:
        # Pass 1: Collect statistics
        state_stats, action_stats, valid_episodes = collect_task_statistics(
            task_id, args.src_path, args.debug
        )
        
        if not valid_episodes:
            print(f"[WARNING] No valid episodes for task {task_id}")
            continue
        
        # Save statistics
        save_stats(args.stats_path, task_id, state_stats, action_stats)
        
        # Pass 2: Normalize and save
        count = process_and_save_episodes(
            task_id, args.src_path, args.output_path,
            state_stats, action_stats, valid_episodes, args.debug
        )
        total_processed += count
    
    print("\n" + "=" * 50)
    print(f"Total episodes processed: {total_processed}")
    print("=" * 50)


if __name__ == "__main__":
    main()
