# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Local file-based action multiview dataset with temporal concatenation.

from __future__ import annotations

import json
import random
import traceback
import warnings
from typing import Sequence

import numpy as np
import torch

from cosmos_predict2._src.predict2.action.datasets.dataset_df import Dataset_3D_DF


class ActionMultiViewDatasetDF(Dataset_3D_DF):
    """Action-conditioned multiview dataset with temporal concatenation.

    Produces video tensors shaped as C x (V*T) x H x W and multiview metadata
    compatible with predict2_multiview pipelines.
    """

    def __init__(
        self,
        *,
        cam_ids: Sequence[int],
        camera_keys: Sequence[str] | None = None,
        **kwargs,
    ):
        if len(cam_ids) < 1:
            raise ValueError("cam_ids must contain at least one camera id")
        self._camera_keys = list(camera_keys) if camera_keys is not None else None
        if self._camera_keys is not None and len(self._camera_keys) != len(cam_ids):
            raise ValueError("camera_keys length must match cam_ids length")

        self._video_size = kwargs.get("video_size", [256, 256])
        super().__init__(cam_ids=list(cam_ids), **kwargs)

    def _get_obs(self, label, frame_ids, cam_id, pre_encode):
        if cam_id is None:
            cam_ids_to_use = self.cam_ids
        else:
            cam_ids_to_use = cam_id

        frames_list = []
        for cid in cam_ids_to_use:
            frames = self._get_frames(label, frame_ids, cam_id=cid, pre_encode=pre_encode)
            frames_list.append(frames)

        return frames_list, list(cam_ids_to_use)

    def _get_camera_keys(self, cam_ids: Sequence[int], label: dict) -> list[str]:
        if self._camera_keys is not None:
            return list(self._camera_keys)
        # Fallback: derive keys from video paths if provided.
        keys = []
        videos = label.get("videos", [])
        for cam_id in cam_ids:
            try:
                video_path = videos[cam_id]["video_path"]
            except Exception:
                keys.append(str(cam_id))
                continue
            # Attempt to extract view name from path segments.
            if "/observation.images." in video_path:
                keys.append(video_path.split("/observation.images.")[-1].split("/")[0])
            else:
                keys.append(str(cam_id))
        return keys

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
            arm_states, gripper_states = self._get_robot_states(label, frame_ids)
            actions = self._get_actions(arm_states, gripper_states, self.accumulate_action)
            actions *= self.c_act_scaler

            data = dict()
            if self.load_action:
                data["action"] = actions.float()

            if self.pre_encode:
                raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
            else:
                frames_list, cam_ids_used = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                # frames_list: List[T, C, H, W] for each view
                video = torch.cat(frames_list, dim=0)  # (V*T, C, H, W)
                video = video.permute(1, 0, 2, 3)  # (C, V*T, H, W)
                data["video"] = video.to(dtype=torch.uint8)

            data["annotation_file"] = ann_file
            data["text"] = text

            # NOTE: __key__ is used to uniquely identify the sample, required for callback functions
            if "episode_index" in label:
                data["__key__"] = label["episode_index"]
            else:
                try:
                    data["__key__"] = label["original_path"]
                except Exception:
                    try:
                        data["__key__"] = label["episode_metadata"]["episode_id"]
                    except Exception:
                        data["__key__"] = label["episode_metadata"]["segment_id"]

            # Multiview metadata
            num_views = len(cam_ids_used)
            num_video_frames_per_view = len(frame_ids)
            view_indices = []
            for view_idx in range(num_views):
                view_indices.extend([view_idx] * num_video_frames_per_view)

            camera_keys_selection = self._get_camera_keys(cam_ids_used, label)
            data["ai_caption"] = [text for _ in range(num_views)]
            data["view_indices"] = torch.tensor(view_indices, dtype=torch.int64)
            data["view_indices_selection"] = torch.tensor(list(range(num_views)), dtype=torch.int64)
            data["sample_n_views"] = torch.tensor(num_views, dtype=torch.int64)
            data["num_video_frames_per_view"] = torch.tensor(num_video_frames_per_view, dtype=torch.int64)
            data["camera_keys_selection"] = camera_keys_selection
            data["front_cam_view_idx_sample_position"] = torch.tensor(0, dtype=torch.int64)
            data["ref_cam_view_idx_sample_position"] = torch.tensor(0, dtype=torch.int64)

            # Just add these to fit the interface
            if self.load_t5_embeddings:
                t5_embeddings = np.squeeze(np.load(ann_file.replace(".json", ".npy")))
                data["t5_text_embeddings"] = torch.from_numpy(t5_embeddings)
            else:
                data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16)
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64)
            data["fps"] = torch.tensor(4.0, dtype=torch.float32)
            data["image_size"] = 256 * torch.ones(4)
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, self._video_size[0], self._video_size[1])

            return data
        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped "
                f"(by randomly sampling another sample in the same dataset)."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            return self[np.random.randint(len(self.samples))]
