# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Preprocess LeRobot v2.1 datasets to JSON annotations for action-conditioned training.

The script is config-driven: add a new entry to `LEROBOT_CONFIGS` and run with
`--config-name` to process another LeRobot dataset.

Usage:
    python scripts/preprocessing/preprocess_lerobot.py --config-name droid
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

LEROBOT_CONFIGS = {
    "droid": {
        "dataset_path": "/mnt/central_storage/data_pool/droid_lerobot",
        "output_path": "datasets/droid/annotation/all",
        "state_key": "observation.state",
        "action_key": "action",
        "camera_views": [
            "observation.images.exterior_image_1_left",
            "observation.images.exterior_image_2_left",
            "observation.images.wrist_image_left",
        ],
    },
}

_WORKER_CONTEXT: dict = {}


def load_task_text_map(tasks_jsonl_path: Path) -> dict[int, str]:
    """Load task text mapping from `meta/tasks.jsonl`."""
    mapping: dict[int, str] = {}
    with open(tasks_jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            mapping[int(record["task_index"])] = str(record.get("task", "") or "")
    return mapping


def to_float_list(value: np.ndarray | list[float]) -> list[float]:
    """Convert a state/action value to a flat Python float list."""
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(x) for x in arr]


def parse_episode_index(parquet_path: Path) -> int:
    """Parse episode index from a parquet file name like `episode_000123.parquet`."""
    stem = parquet_path.stem
    if not stem.startswith("episode_"):
        raise ValueError(f"Invalid parquet filename: {parquet_path.name}")
    return int(stem.replace("episode_", ""))


def build_video_entries(
    dataset_path: Path,
    chunk_name: str,
    episode_index: int,
    camera_views: list[str],
) -> tuple[list[dict[str, str]], list[str]]:
    """
    Build video entries for one episode.

    Returns:
        - list of {"video_path": "..."} entries in camera order
        - list of missing video paths for validation/reporting
    """
    entries: list[dict[str, str]] = []
    missing: list[str] = []
    episode_filename = f"episode_{episode_index:06d}.mp4"
    for view in camera_views:
        video_path = dataset_path / "videos" / chunk_name / view / episode_filename
        if not video_path.exists():
            missing.append(str(video_path))
        entries.append({"video_path": str(video_path)})
    return entries, missing


def process_episode(
    parquet_path: Path,
    dataset_path: Path,
    state_key: str,
    action_key: str,
    camera_views: list[str],
    task_text_map: dict[int, str],
    strict_missing_videos: bool,
) -> tuple[dict, int]:
    """Convert one parquet episode into output JSON dict and missing-video count."""
    required_columns = [state_key, action_key, "task_index", "episode_index"]
    df = pd.read_parquet(parquet_path, columns=required_columns)
    if df.empty:
        raise ValueError(f"Empty parquet: {parquet_path}")

    episode_index_from_file = parse_episode_index(parquet_path)
    episode_index_from_data = int(df["episode_index"].iloc[0])
    if episode_index_from_file != episode_index_from_data:
        raise ValueError(
            f"Episode index mismatch for {parquet_path}: "
            f"filename={episode_index_from_file}, parquet={episode_index_from_data}"
        )

    task_index = int(df["task_index"].iloc[0])
    text = task_text_map.get(task_index, "")

    chunk_name = parquet_path.parent.name
    videos, missing_videos = build_video_entries(
        dataset_path=dataset_path,
        chunk_name=chunk_name,
        episode_index=episode_index_from_file,
        camera_views=camera_views,
    )
    if strict_missing_videos and missing_videos:
        raise FileNotFoundError(
            f"Missing {len(missing_videos)} videos for episode {episode_index_from_file}: "
            f"{missing_videos}"
        )

    states = [to_float_list(v) for v in df[state_key].tolist()]
    actions = [to_float_list(v) for v in df[action_key].tolist()]
    if len(states) != len(actions):
        raise ValueError(
            f"State/action length mismatch for episode {episode_index_from_file}: "
            f"{len(states)} vs {len(actions)}"
        )

    output = {
        "text": text,
        "videos": videos,
        "state": states,
        "action": actions,
        "episode_index": episode_index_from_file,
        "timesteps": len(df),
    }
    return output, len(missing_videos)


def _init_worker(
    dataset_path: str,
    output_path: str,
    state_key: str,
    action_key: str,
    camera_views: list[str],
    task_text_map: dict[int, str],
    strict_missing_videos: bool,
):
    """Initialize per-process immutable context."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "dataset_path": Path(dataset_path),
        "output_path": Path(output_path),
        "state_key": state_key,
        "action_key": action_key,
        "camera_views": camera_views,
        "task_text_map": task_text_map,
        "strict_missing_videos": strict_missing_videos,
    }


def _process_and_save_episode_worker(parquet_path_str: str) -> tuple[bool, int, str, str]:
    """
    Worker entrypoint: process one episode parquet and write output JSON.

    Returns:
        (ok, missing_video_count, parquet_path, error_message)
    """
    parquet_path = Path(parquet_path_str)
    try:
        output, missing_count = process_episode(
            parquet_path=parquet_path,
            dataset_path=_WORKER_CONTEXT["dataset_path"],
            state_key=_WORKER_CONTEXT["state_key"],
            action_key=_WORKER_CONTEXT["action_key"],
            camera_views=_WORKER_CONTEXT["camera_views"],
            task_text_map=_WORKER_CONTEXT["task_text_map"],
            strict_missing_videos=_WORKER_CONTEXT["strict_missing_videos"],
        )
        save_path = _WORKER_CONTEXT["output_path"] / f"{output['episode_index']:06d}.json"
        with open(save_path, "w") as f:
            json.dump(output, f, indent=4)
        return True, missing_count, parquet_path_str, ""
    except Exception as e:
        return False, 0, parquet_path_str, str(e)


def main():
    parser = argparse.ArgumentParser(description="Preprocess LeRobot dataset to JSON annotations")
    parser.add_argument(
        "--config-name",
        default="droid",
        choices=sorted(LEROBOT_CONFIGS.keys()),
        help="Dataset config name from LEROBOT_CONFIGS",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Override output directory (defaults to config output_path)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on number of episodes to process (for smoke/debug)",
    )
    parser.add_argument(
        "--strict-missing-videos",
        action="store_true",
        help="Fail if any expected camera video is missing for an episode",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, min(32, os.cpu_count() or 1)),
        help="Number of worker processes. Set to 1 for single-process mode.",
    )
    parser.add_argument(
        "--worker-chunksize",
        type=int,
        default=16,
        help="Chunk size passed to multiprocessing imap_unordered.",
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        raise ValueError(f"--num-workers must be >= 1, got {args.num_workers}")
    if args.worker_chunksize < 1:
        raise ValueError(f"--worker-chunksize must be >= 1, got {args.worker_chunksize}")

    config = LEROBOT_CONFIGS[args.config_name]
    dataset_path = Path(config["dataset_path"])
    output_path = args.output_path if args.output_path is not None else Path(config["output_path"])
    state_key = config["state_key"]
    action_key = config["action_key"]
    camera_views = config["camera_views"]

    data_root = dataset_path / "data"
    meta_tasks_path = dataset_path / "meta" / "tasks.jsonl"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")
    if not meta_tasks_path.exists():
        raise FileNotFoundError(f"tasks.jsonl not found: {meta_tasks_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(data_root.glob("chunk-*/episode_*.parquet"))
    if args.max_episodes is not None:
        parquet_files = parquet_files[: args.max_episodes]

    print(f"Config name: {args.config_name}")
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"State key: {state_key}")
    print(f"Action key: {action_key}")
    print(f"Camera views: {camera_views}")
    print(f"Strict missing videos: {args.strict_missing_videos}")
    print(f"Num workers: {args.num_workers}")
    print(f"Worker chunksize: {args.worker_chunksize}")
    print(f"Episodes to process: {len(parquet_files)}")
    print("-" * 60)

    task_text_map = load_task_text_map(meta_tasks_path)

    processed = 0
    failed = 0
    missing_video_total = 0

    parquet_paths_str = [str(p) for p in parquet_files]
    worker_init_args = (
        str(dataset_path),
        str(output_path),
        state_key,
        action_key,
        camera_views,
        task_text_map,
        args.strict_missing_videos,
    )

    def consume_results(iterator):
        nonlocal processed, failed, missing_video_total
        for ok, missing_count, parquet_path_str, error_msg in tqdm(
            iterator,
            total=len(parquet_paths_str),
            desc="Processing episodes",
        ):
            if ok:
                processed += 1
                missing_video_total += missing_count
            else:
                failed += 1
                print(f"[ERROR] {parquet_path_str}: {error_msg}")

    if args.num_workers <= 1:
        _init_worker(*worker_init_args)
        consume_results(map(_process_and_save_episode_worker, parquet_paths_str))
    else:
        try:
            with Pool(
                processes=args.num_workers,
                initializer=_init_worker,
                initargs=worker_init_args,
            ) as pool:
                consume_results(
                    pool.imap_unordered(
                        _process_and_save_episode_worker,
                        parquet_paths_str,
                        chunksize=args.worker_chunksize,
                    )
                )
        except PermissionError as e:
            print(
                "[WARNING] Failed to start multiprocessing workers "
                f"({e}). Falling back to ThreadPoolExecutor."
            )
            _init_worker(*worker_init_args)
            with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                consume_results(executor.map(_process_and_save_episode_worker, parquet_paths_str))

    print("\n" + "=" * 60)
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Total missing video entries observed: {missing_video_total}")
    print("=" * 60)


if __name__ == "__main__":
    main()
