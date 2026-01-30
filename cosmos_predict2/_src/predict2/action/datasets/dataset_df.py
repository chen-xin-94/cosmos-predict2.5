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
Dataset classes for Data Foundry (DF) action-conditioned video generation.

Includes:
- Dataset_3D_DF: Base DF dataset
- ActionConditionedMultiViewDataset_DF: Multi-view variant
"""

import json
import os
import random
import time
import traceback
import warnings
import subprocess
import imageio
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.utils.dataset_utils import (
    Resize_Preprocess,
    ToTensorVideo,
    quat2rotm,
    rotm2euler,
)
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D


class Dataset_3D_DF(Dataset_3D):
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
        gripper_key="continuous_gripper_state",
        text_key="text",
        gripper_rescale_factor=1.0,
        is_rollout=None,
    ):
        """Dataset class for loading 3D robot action-conditional data with text descriptions.

        Extends Dataset_3D to support quaternion-based arm states and text embeddings.
        Uses FFmpeg for video loading to handle AV1 codec efficiently.

        Args:
            text_key (str): Key to extract text from annotation files. Defaults to "text".
            [Other args inherited from Dataset_3D - see parent class documentation]
        """
        self._text_key = text_key
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
            gripper_key=gripper_key,
            gripper_rescale_factor=gripper_rescale_factor,
            is_rollout=is_rollout,
        )

    def _load_video(self, video_path, frame_ids, fps=30):
        """
        Load specific frames from a video using FFmpeg batch extraction.
        Only decodes the required frames -> stable for AV1.
        Returns numpy array of (T, H, W, C) with frames in the order of frame_ids.
        """
        
        # Sort frame IDs for batch extraction (but remember original order)
        frame_ids_sorted = sorted(frame_ids)
        
        # Probe video to get resolution using ffprobe CLI
        probe_cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_streams',
            video_path
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        probe_data = json.loads(probe_result.stdout)
        video_stream = next(s for s in probe_data['streams'] if s['codec_type'] == 'video')
        W = int(video_stream['width'])
        H = int(video_stream['height'])
        
        # Build FFmpeg select filter to extract specific frames
        # eq(n,5) means "select frame number 5"
        # Join with + to select multiple frames
        select_expr = '+'.join([f'eq(n\\,{fid})' for fid in frame_ids_sorted])
        
        # Single FFmpeg command to extract all requested frames
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'select={select_expr}',  # Select only these frames
            '-vsync', '0',                    # Don't duplicate/drop frames
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            'pipe:1'
        ]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        
        if process.returncode != 0 or not out:
            raise RuntimeError(f"Failed to decode frames {frame_ids_sorted} from {video_path}")
        
        # Reshape raw bytes -> (T, H, W, 3) in sorted order
        num_frames = len(frame_ids)
        frames = np.frombuffer(out, dtype=np.uint8).reshape(num_frames, H, W, 3)
        
        # Reorder frames to match original frame_ids order (if not already sorted)
        if frame_ids != frame_ids_sorted:
            reorder_idx = [frame_ids_sorted.index(fid) for fid in frame_ids]
            frames = frames[reorder_idx]
        
        return frames
    
    def _get_frames(self, label, frame_ids, cam_id, pre_encode):
        """Override to not use os.path.join with self.video_path."""
        if pre_encode:
            raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
        else:
            video_path = label["videos"][cam_id]["video_path"]
            # Note: video_path is already absolute for this dataset
            frames = self._load_video(video_path, frame_ids)
            frames = frames.astype(np.uint8)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)

            def printvideo(videos, filename):
                t_videos = rearrange(videos, "f c h w -> f h w c")
                t_videos = (
                    ((t_videos / 2.0 + 0.5).clamp(0, 1) * 255).detach().to(dtype=torch.uint8).cpu().contiguous().numpy()
                )
                print(t_videos.shape)
                writer = imageio.get_writer(filename, fps=4)
                for frame in t_videos:
                    writer.append_data(frame)

            if self.normalize:
                frames = self.preprocess(frames)
            else:
                frames = self.not_norm_preprocess(frames)
                frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames

    def _get_text(self, label):
        return label[self._text_key]

    def _get_robot_states(self, label, frame_ids):
        """Override to use 7D quaternion states instead of 6D euler angles."""
        all_states = np.array(label[self._state_key])
        all_cont_gripper_states = np.array(label[self._gripper_key])
        states = all_states[frame_ids]
        cont_gripper_states = all_cont_gripper_states[frame_ids]
        arm_states = states[:, :7]
        assert arm_states.shape[0] == self.sequence_length
        assert cont_gripper_states.shape[0] == self.sequence_length
        return arm_states, cont_gripper_states

    def _get_all_actions(self, arm_states, gripper_states, accumulate_action):
        action_num = arm_states.shape[0] - 1
        action = np.zeros((action_num, self.action_dim))
        if accumulate_action:
            first_xyz = arm_states[0, 0:3]
            first_quat = arm_states[0, 3:7]
            first_rotm = quat2rotm(first_quat)
            for k in range(1, action_num + 1):
                curr_xyz = arm_states[k, 0:3]
                curr_quat = arm_states[k, 3:7]
                curr_gripper = gripper_states[k]
                curr_rotm = quat2rotm(curr_quat)
                rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
                rel_rotm = first_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        else:
            for k in range(1, action_num + 1):
                prev_xyz = arm_states[k - 1, 0:3]
                prev_quat = arm_states[k - 1, 3:7]
                prev_rotm = quat2rotm(prev_quat)
                curr_xyz = arm_states[k, 0:3]
                curr_quat = arm_states[k, 3:7]
                curr_gripper = gripper_states[k]
                curr_rotm = quat2rotm(curr_quat)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)

    def _get_actions(self, arm_states, gripper_states, accumulate_action):
        action = np.zeros((self.sequence_length - 1, self.action_dim))
        if accumulate_action:
            base_xyz = arm_states[0, 0:3]
            base_quat = arm_states[0, 3:7]
            base_rotm = quat2rotm(base_quat)
            for k in range(1, self.sequence_length):
                curr_xyz = arm_states[k, 0:3]
                curr_quat = arm_states[k, 3:7]
                curr_gripper = gripper_states[k]
                curr_rotm = quat2rotm(curr_quat)
                rel_xyz = np.dot(base_rotm.T, curr_xyz - base_xyz)
                rel_rotm = base_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
                if k % 4 == 0:
                    base_xyz = arm_states[k, 0:3]
                    base_quat = arm_states[k, 3:7]
                    base_rotm = quat2rotm(base_quat)
        else:
            for k in range(1, self.sequence_length):
                prev_xyz = arm_states[k - 1, 0:3]
                prev_quat = arm_states[k - 1, 3:7]
                prev_rotm = quat2rotm(prev_quat)
                curr_xyz = arm_states[k, 0:3]
                curr_quat = arm_states[k, 3:7]
                curr_gripper = gripper_states[k]
                curr_rotm = quat2rotm(curr_quat)
                rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
                rel_rotm = prev_rotm.T @ curr_rotm
                rel_rpy = rotm2euler(rel_rotm)
                action[k - 1, 0:3] = rel_xyz
                action[k - 1, 3:6] = rel_rpy
                action[k - 1, 6] = curr_gripper
        return torch.from_numpy(action)  # (l - 1, act_dim)

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
                video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
                video = video.permute(1, 0, 2, 3)  # Rearrange from [T, C, H, W] to [C, T, H, W]
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

            # Just add these to fit the interface
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


class ActionConditionedMultiViewDataset_DF(Dataset_3D_DF):
    """Multi-view variant of Dataset_3D_DF supporting 3+ camera views.
    
    This class extends Dataset_3D_DF to concatenate multiple camera views
    along the width dimension. Unlike ActionConditionedMultiViewDataset which
    is limited to 2 views with a specific selection pattern, this class
    supports an arbitrary number of views specified in cam_ids.
    
    Args:
        cam_ids: List of integer indices corresponding to entries in the
            JSON "videos" list. Example: [0, 1, 2] for all 3 views.
    
    Example usage:
        dataset = ActionConditionedMultiViewDataset_DF(
            ...,
            cam_ids=[0, 1, 2],  # All 3 views concatenated
            ...
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
            Tuple of (frames, cam_ids_used) where frames has shape (L, C, H, W*num_views)
        """
        if cam_id is None:
            cam_ids_to_use = self.cam_ids
        else:
            cam_ids_to_use = cam_id
        
        frames_list = []
        for cid in cam_ids_to_use:
            frames = self._get_frames(label, frame_ids, cam_id=cid, pre_encode=pre_encode)
            frames_list.append(frames)
        
        # Concatenate all views along width (dim=3 for L,C,H,W format)
        combined_frames = torch.cat(frames_list, dim=3)
        return combined_frames, cam_ids_to_use
