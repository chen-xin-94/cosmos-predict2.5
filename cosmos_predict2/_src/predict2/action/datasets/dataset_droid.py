# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dataset classes for DROID action-conditioned video generation.

Key behaviors:
- Load preprocessed actions directly from JSON (no state->action conversion)
- Normalize actions with stats from assets/action_conditioned/concat_view/droid/stats.json
- Support multi-view image-space concatenation across camera views
"""

import json
import os
import random
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch

from cosmos_predict2._src.predict2.action.datasets.dataset_df import Dataset_3D_DF


class Dataset_3D_DROID(Dataset_3D_DF):
    """Dataset class for loading DROID action-conditioned data."""

    def __init__(
        self,
        train_annotation_path,
        val_annotation_path,
        test_annotation_path,
        video_path,
        fps_downsample_ratio,
        num_action_per_chunk,
        cam_ids,
        accumulate_action,
        video_size,
        val_start_frame_interval,
        debug=False,
        normalize=False,
        pre_encode=False,
        do_evaluate=False,
        load_t5_embeddings=False,
        load_action=True,
        mode="train",
        state_key="state",
        text_key="text",
        action_key="action",
        action_stats_path="assets/action_conditioned/concat_view/droid/stats.json",
        action_normalization="minmax",
    ):
        self._text_key = text_key
        self._action_key = action_key
        self._action_normalization = action_normalization
        self._action_min = None
        self._action_max = None
        self._action_mean = None
        self._action_std = None

        super().__init__(
            train_annotation_path=train_annotation_path,
            val_annotation_path=val_annotation_path,
            test_annotation_path=test_annotation_path,
            video_path=video_path,
            fps_downsample_ratio=fps_downsample_ratio,
            num_action_per_chunk=num_action_per_chunk,
            cam_ids=cam_ids,
            accumulate_action=accumulate_action,
            video_size=video_size,
            val_start_frame_interval=val_start_frame_interval,
            debug=debug,
            normalize=normalize,
            pre_encode=pre_encode,
            do_evaluate=do_evaluate,
            load_t5_embeddings=load_t5_embeddings,
            load_action=load_action,
            mode=mode,
            state_key=state_key,
            gripper_key="continuous_gripper_state",
            text_key=text_key,
            gripper_rescale_factor=1.0,
            is_rollout=None,
        )

        self.action_dim = 7
        self.c_act_scaler = 1.0
        self._load_action_stats(action_stats_path)

    def _load_action_stats(self, action_stats_path: str) -> None:
        stats_path = Path(action_stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"Action stats file not found: {stats_path}")

        with open(stats_path, "r") as f:
            stats = json.load(f)

        action_stats = stats["action"]
        self._action_min = np.asarray(action_stats["min"], dtype=np.float32)
        self._action_max = np.asarray(action_stats["max"], dtype=np.float32)
        self._action_mean = np.asarray(action_stats["mean"], dtype=np.float32)
        self._action_std = np.asarray(action_stats["std"], dtype=np.float32)

    def _normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        if self._action_normalization in ("none", None):
            return actions

        if self._action_normalization == "minmax":
            denom = self._action_max - self._action_min
            denom = np.where(denom == 0, 1.0, denom)
            return np.clip((actions - self._action_min) / denom, 0.0, 1.0)

        if self._action_normalization == "standard":
            std = np.where(self._action_std == 0, 1.0, self._action_std)
            return (actions - self._action_mean) / std

        raise ValueError(f"Unsupported action_normalization={self._action_normalization}")

    def _get_text(self, label):
        return label.get(self._text_key, "")

    def __getitem__(self, index, cam_id=None, return_video=False):
        if self.mode != "train":
            np.random.seed(index)
            random.seed(index)

        try:
            sample = self.samples[index]
            ann_file = sample["ann_file"]
            frame_ids = sample["frame_ids"]
            with open(ann_file, "r") as f:
                label = json.load(f)

            text = self._get_text(label)
            data = dict()

            if self.load_action:
                all_actions = np.asarray(label[self._action_key], dtype=np.float32)
                action_frame_ids = frame_ids[1:]
                actions = all_actions[action_frame_ids]
                if actions.shape[1] != self.action_dim:
                    raise ValueError(
                        f"Expected action dim {self.action_dim}, got {actions.shape[1]} in {ann_file}"
                    )
                actions = self._normalize_actions(actions)
                data["action"] = torch.from_numpy(actions).float()

            if self.pre_encode:
                raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")

            video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
            video = video.permute(1, 0, 2, 3)
            data["video"] = video.to(dtype=torch.uint8)

            data["annotation_file"] = ann_file
            data["text"] = text
            data["__key__"] = str(label.get("episode_index", os.path.basename(ann_file).replace(".json", "")))

            if self.load_t5_embeddings:
                t5_embeddings = np.squeeze(np.load(ann_file.replace(".json", ".npy")))
                data["t5_text_embeddings"] = torch.from_numpy(t5_embeddings)
            else:
                data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16)
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64)
            data["fps"] = 4
            data["image_size"] = 256 * torch.ones(4)
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 256, 256)
            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]


class ActionConditionedMultiViewDataset_DROID(Dataset_3D_DROID):
    """Multi-view variant of Dataset_3D_DROID with width concatenation."""

    def _get_obs(self, label, frame_ids, cam_id, pre_encode):
        if cam_id is None:
            cam_ids_to_use = self.cam_ids
        else:
            cam_ids_to_use = cam_id

        frames_list = []
        for cid in cam_ids_to_use:
            frames = self._get_frames(label, frame_ids, cam_id=cid, pre_encode=pre_encode)
            frames_list.append(frames)

        combined_frames = torch.cat(frames_list, dim=3)
        return combined_frames, cam_ids_to_use
