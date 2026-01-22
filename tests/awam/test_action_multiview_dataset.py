#!/usr/bin/env python3
"""Smoke tests for action multiview dataset."""

import json
import os
from pathlib import Path

import pytest

from cosmos_predict2._src.predict2_multiview.datasets.action_local import ActionMultiViewDatasetDF


def _example_base_path() -> Path:
    return Path(
        "datasets_example/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand"
    )


def test_action_multiview_dataset_shapes():
    base_path = _example_base_path()
    ann_path = base_path / "annotation" / "train"
    ann_files = sorted(ann_path.glob("*.json"))
    if not ann_files:
        pytest.skip("No annotation files found for dataset_example")

    with ann_files[0].open() as f:
        label = json.load(f)
    video_path = label["videos"][0]["video_path"]
    if not os.path.exists(video_path):
        pytest.skip(f"Video file not found: {video_path}")
    if video_path.startswith("/mnt/central_storage") and os.environ.get("AWAM_RUN_DATASET_TESTS") != "1":
        pytest.skip("Skipping large dataset read; set AWAM_RUN_DATASET_TESTS=1 to run.")

    dataset = ActionMultiViewDatasetDF(
        train_annotation_path=str(ann_path),
        val_annotation_path=str(base_path / "annotation" / "val"),
        test_annotation_path=str(base_path / "annotation" / "test"),
        video_path="",
        fps_downsample_ratio=1,
        num_action_per_chunk=12,
        cam_ids=[0, 1, 2],
        camera_keys=["frame_camera_left", "frame_camera_top", "wrist_camera"],
        accumulate_action=False,
        video_size=[256, 320],
        val_start_frame_interval=100,
        debug=True,
        normalize=False,
        mode="train",
        state_key="state",
        gripper_key="continuous_gripper_state",
        text_key="text",
    )

    sample = dataset[0]
    video = sample["video"]
    num_views = 3
    expected_frames = 13 * num_views

    assert video.shape[1] == expected_frames
    assert sample["view_indices"].shape[0] == expected_frames
    assert sample["sample_n_views"].item() == num_views
