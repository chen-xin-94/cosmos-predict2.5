# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Dataset classes for AgiBotWorld action-conditioned video generation.

Key differences from Dataset_3D / Dataset_3D_DF:
- Actions are loaded directly from JSON (36-dim), no transformation needed
- No gripper state handling
- Multi-task support: traverses all task folders under base_annotation_path
- Actions are pre-scaled to [0, 1], no scaler applied

Run this command to interactively debug:
PYTHONPATH=. python cosmos_predict2/_src/predict2/action/datasets/dataset_agibot.py
"""

import json
import os
import random
import traceback
import warnings

import numpy as np
import torch

from cosmos_predict2._src.predict2.action.datasets.dataset_df import Dataset_3D_DF


class Dataset_3D_AGIBOT(Dataset_3D_DF):
    """Dataset class for loading AgiBotWorld action-conditioned data.

    This dataset extends Dataset_3D_DF to support AgiBotWorld-specific features:
    - Pre-computed 36-dimensional actions loaded directly from JSON
    - Multi-task support: traverses all task folders under base_annotation_path
    - Actions are pre-scaled to [0, 1], no scaler applied

    Args:
        base_annotation_path (str): Base path to annotation files (e.g., "datasets/agibot/annotation")
            Will traverse all task subdirectories (task_*/train, task_*/val, task_*/test)
        video_path (str): Base path to video files (unused for AgiBotWorld, paths are absolute in JSON)
        fps_downsample_ratio (int or list): Interval between sampled frames in a sequence
        num_action_per_chunk (int): Number of actions to load per sequence (sequence_length = 1 + num_action_per_chunk)
        cam_ids (list): List of camera indices to use (e.g., [0] for single view)
        accumulate_action (bool): Whether to accumulate actions (not used for AgiBotWorld)
        video_size (list): Target size [H, W] for video frames
        val_start_frame_interval (int): Frame sampling interval for validation/test
        debug (bool): If True, only loads subset of data
        normalize (bool): Whether to normalize video frames
        pre_encode (bool): Whether to use pre-encoded features (not supported)
        do_evaluate (bool): Whether in evaluation mode
        load_t5_embeddings (bool): Whether to load T5 embeddings
        load_action (bool): Whether to load actions
        mode (str): Dataset mode - 'train', 'val' or 'test'
        state_key (str): Key to access state in JSON (default: "state")
        text_key (str): Key to access text in JSON (default: "text")
    """

    def __init__(
        self,
        base_annotation_path,
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
    ):
        self.base_annotation_path = base_annotation_path
        
        # Call parent constructor
        super().__init__(
            train_annotation_path=base_annotation_path,  # Will be overridden by _init_anns
            val_annotation_path=base_annotation_path,
            test_annotation_path=base_annotation_path,
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
            gripper_key="continuous_gripper_state",  # Not used but required by parent
            text_key=text_key,
            gripper_rescale_factor=1.0,
            is_rollout=None,
        )
        
        # Override action dimension AFTER parent init (parent sets action_dim=7 by default)
        self.action_dim = 36  # AgiBotWorld action dimension
        # No action scaling for AGIBOT (actions already in [0,1])
        self.c_act_scaler = 1.0

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.base_annotation_path}"

    def _init_anns(self, data_dir):
        """Traverse all task directories and collect annotation files.
        
        Structure: base_path/task_*/[train|val|test]/*.json
        
        Args:
            data_dir: Base annotation path (ignored, uses self.base_annotation_path)
        """
        ann_files = []
        base_path = self.base_annotation_path
        
        # List all task directories
        if not os.path.exists(base_path):
            raise ValueError(f"Base annotation path does not exist: {base_path}")
        
        task_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        
        for task_dir in task_dirs:
            task_path = os.path.join(base_path, task_dir)
            mode_path = os.path.join(task_path, self.mode)
            
            if os.path.exists(mode_path):
                json_files = [
                    os.path.join(mode_path, f)
                    for f in os.listdir(mode_path)
                    if f.endswith(".json")
                ]
                ann_files.extend(json_files)
        
        if len(ann_files) == 0:
            warnings.warn(f"No annotation files found in {base_path} for mode {self.mode}")
        
        return ann_files

    def _get_actions(self, arm_states, gripper_states, accumulate_action):
        """Load actions directly from JSON annotation (called by parent's __getitem__).
        
        Note: This override changes the signature semantics - arm_states is actually
        the label dict, and gripper_states/accumulate_action are ignored.
        
        AgiBotWorld actions are pre-computed 36-dim vectors already scaled to [0, 1].
        We return actions for frames 1 to sequence_length (not the first frame).
        """
        # HACK: Parent __getitem__ calls this with (arm_states, gripper_states, accumulate_action)
        # but we need (label, frame_ids). We'll need to override __getitem__ to fix this properly.
        # For now, this won't be called - see overridden __getitem__ below.
        raise NotImplementedError(
            "This method should not be called directly. "
            "Use the overridden __getitem__ which loads actions from JSON."
        )

    def __getitem__(self, index, cam_id=None, return_video=False):
        """Override to load actions directly from JSON instead of computing from states."""
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
                # Load actions directly from JSON (AgiBotWorld-specific)
                all_actions = np.array(label["action"])
                action_frame_ids = frame_ids[1:]  # Actions for frames 1 to sequence_length
                actions = all_actions[action_frame_ids]
                
                assert actions.shape == (self.sequence_length - 1, self.action_dim), (
                    f"Expected actions shape ({self.sequence_length - 1}, {self.action_dim}), "
                    f"got {actions.shape}"
                )
                data["action"] = torch.from_numpy(actions).float()

            if self.pre_encode:
                raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
            else:
                video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
                data["video"] = video.to(dtype=torch.uint8)

            data["annotation_file"] = ann_file
            data["text"] = text

            # NOTE: __key__ is used to uniquely identify the sample, required for callback functions
            if "episode_index" in label:
                data["__key__"] = str(label["episode_index"])
            else:
                data["__key__"] = os.path.basename(ann_file).replace(".json", "")

            # Add placeholders to fit the interface
            if self.load_t5_embeddings:
                t5_embeddings = np.squeeze(np.load(ann_file.replace(".json", ".npy")))
                data["t5_text_embeddings"] = torch.from_numpy(t5_embeddings)
            else:
                data["t5_text_embeddings"] = torch.zeros(512, 1024, dtype=torch.bfloat16)
                data["ai_caption"] = ""
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


class ActionConditionedMultiViewDataset_AGIBOT(Dataset_3D_AGIBOT):
    """Multi-view variant of Dataset_3D_AGIBOT supporting multiple camera views.
    
    This class extends Dataset_3D_AGIBOT to concatenate multiple camera views
    along the width dimension.
    
    Args:
        cam_ids: List of integer indices corresponding to entries in the
            JSON "videos" list. Example: [0, 1, 2] for all 3 views.
    
    Example usage:
        dataset = ActionConditionedMultiViewDataset_AGIBOT(
            ...,
            cam_ids=[0, 1, 2],  # All 3 views: hand_left, head, hand_right
            video_size=[480, 640],  # Per-view size, concatenated to 480x1920
            ...,
        )
    """

    def _get_obs(self, label, frame_ids, cam_id, pre_encode):
        """Override to load and concatenate multiple camera views.
        
        Args:
            label: JSON annotation dict containing "videos" list
            frame_ids: List of frame indices to load
            cam_id: Optional override for camera IDs (uses self.cam_ids if None)
            pre_encode: Whether to use pre-encoded features (not supported)
        
        Returns:
            Tuple of (frames, cam_ids_used) where frames has shape (T, C, H, W*num_views)
        """
        if cam_id is None:
            cam_ids_to_use = self.cam_ids
        else:
            cam_ids_to_use = cam_id
        
        frames_list = []
        for cid in cam_ids_to_use:
            frames = self._get_frames(label, frame_ids, cam_id=cid, pre_encode=pre_encode)
            frames_list.append(frames)
        
        # Concatenate all views along width (dim=3 for T,C,H,W format)
        combined_frames = torch.cat(frames_list, dim=3)
        return combined_frames, cam_ids_to_use


if __name__ == "__main__":
    """
    PYTHONPATH=. python cosmos_predict2/_src/predict2/action/datasets/dataset_agibot.py
    """
    # Test single-view dataset
    print("Testing Dataset_3D_AGIBOT...")
    dataset = Dataset_3D_AGIBOT(
        base_annotation_path="datasets/agibot/annotation",
        video_path="",
        fps_downsample_ratio=6,
        num_action_per_chunk=12,
        cam_ids=[0],
        accumulate_action=False,
        video_size=[480, 640],
        val_start_frame_interval=1,
        mode="train",
    )
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Video shape: {sample['video'].shape}")  # Expected: (3, 13, 480, 640)
        print(f"Action shape: {sample['action'].shape}")  # Expected: (12, 36)
        print(f"Text: {sample['text']}")
    
    # Test multi-view dataset
    print("\nTesting ActionConditionedMultiViewDataset_AGIBOT...")
    mv_dataset = ActionConditionedMultiViewDataset_AGIBOT(
        base_annotation_path="datasets/agibot/annotation",
        video_path="",
        fps_downsample_ratio=6,
        num_action_per_chunk=12,
        cam_ids=[0, 1, 2],
        accumulate_action=False,
        video_size=[480, 640],
        val_start_frame_interval=1,
        mode="train",
    )
    print(f"Multi-view dataset size: {len(mv_dataset)}")
    
    if len(mv_dataset) > 0:
        sample = mv_dataset[0]
        print(f"Video shape: {sample['video'].shape}")  # Expected: (3, 13, 480, 1920)
        print(f"Action shape: {sample['action'].shape}")  # Expected: (12, 36)

    from IPython import embed
    embed()
