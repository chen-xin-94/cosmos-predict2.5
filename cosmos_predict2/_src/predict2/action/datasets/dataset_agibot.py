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
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import ffmpeg
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from tqdm import tqdm

from cosmos_predict2._src.imaginaire.utils.dataset_utils import (
    Resize_Preprocess,
    ToTensorVideo,
)


class Dataset_3D_AGIBOT(Dataset):
    """Dataset class for loading AgiBotWorld action-conditioned data.

    This dataset loads robot trajectories consisting of RGB video frames and
    pre-computed 36-dimensional actions directly from JSON annotations.

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
        super().__init__()
        self.base_annotation_path = base_annotation_path
        self.video_path = video_path
        self.fps_downsample_ratio = fps_downsample_ratio
        self.mode = mode

        if mode == "train":
            self.start_frame_interval = 1
        else:
            self.start_frame_interval = val_start_frame_interval

        self.sequence_length = 1 + num_action_per_chunk
        self.normalize = normalize
        self.pre_encode = pre_encode
        self.load_t5_embeddings = load_t5_embeddings
        self.load_action = load_action

        self.cam_ids = cam_ids
        self.accumulate_action = accumulate_action

        self.action_dim = 36  # AgiBotWorld action dimension
        self._state_key = state_key
        self._text_key = text_key

        # Collect annotation files from all task directories
        self.ann_files = self._init_anns(self.base_annotation_path)
        print(f"{len(self.ann_files)} trajectories in total")

        self.samples = self._init_sequences(self.ann_files)
        self.samples = sorted(self.samples, key=lambda x: (x["ann_file"], x["frame_ids"][0]))

        if debug and not do_evaluate:
            self.samples = self.samples[0:10]

        print(f"{len(self.ann_files)} trajectories in total")
        print(f"{len(self.samples)} samples in total")

        self.wrong_number = 0
        self.transform = T.Compose([T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)])
        self.training = False
        self.preprocess = T.Compose(
            [
                ToTensorVideo(),
                Resize_Preprocess(tuple(video_size)),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.not_norm_preprocess = T.Compose([ToTensorVideo(), Resize_Preprocess(tuple(video_size))])

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.base_annotation_path}"

    def _init_anns(self, base_path):
        """Traverse all task directories and collect annotation files.
        
        Structure: base_path/task_*/[train|val|test]/*.json
        """
        ann_files = []
        
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

    def _init_sequences(self, ann_files):
        samples = []
        with ThreadPoolExecutor(32) as executor:
            future_to_ann_file = {
                executor.submit(self._load_and_process_ann_file, ann_file): ann_file for ann_file in ann_files
            }
            for future in tqdm(as_completed(future_to_ann_file), total=len(ann_files)):
                samples.extend(future.result())
        return samples

    def _load_and_process_ann_file(self, ann_file):
        samples = []
        with open(ann_file, "r") as f:
            ann = json.load(f)

        n_frames = len(ann["action"])  # Use action length as reference

        if isinstance(self.fps_downsample_ratio, int):
            fps_downsample_ratio_list = [self.fps_downsample_ratio]
        else:
            fps_downsample_ratio_list = self.fps_downsample_ratio

        for fps_downsample_ratio in fps_downsample_ratio_list:
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
                    curr_frame_i += fps_downsample_ratio
                # make sure there are sequence_length number of frames
                if len(sample["frame_ids"]) == self.sequence_length:
                    samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_video(self, video_path, frame_ids, fps=30):
        """
        Load specific frames from a video using FFmpeg.
        Only decodes the required frames -> stable for various codecs.
        Returns numpy array of (T, H, W, C).
        """
        frames = []
        frame_ids_sorted = sorted(frame_ids)
        H, W = None, None

        for fid in frame_ids_sorted:
            # Convert frame index -> timestamp in seconds
            ts = fid / fps

            # FFmpeg command to extract ONE frame
            process = (
                ffmpeg
                .input(video_path, ss=ts)
                .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
                .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
            )

            # Read raw frame bytes
            out = process.stdout.read()
            process.stdout.close()
            process.wait()

            if not out:
                raise RuntimeError(f"Failed to decode frame {fid} @ {ts}s from {video_path}")

            # Determine resolution by probing once
            if H is None:
                probe = ffmpeg.probe(video_path)
                video_stream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
                W = int(video_stream['width'])
                H = int(video_stream['height'])

            # Reshape raw buffer -> (H, W, 3)
            frame = np.frombuffer(out, np.uint8).reshape(H, W, 3)
            frames.append(frame)

        return np.stack(frames, axis=0)

    def _get_frames(self, label, frame_ids, cam_id, pre_encode):
        if pre_encode:
            raise NotImplementedError("Pre-encoded videos are not supported for this dataset.")
        else:
            video_path = label["videos"][cam_id]["video_path"]
            # video_path is absolute for AgiBotWorld
            frames = self._load_video(video_path, frame_ids)
            frames = frames.astype(np.uint8)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (l, c, h, w)

            if self.normalize:
                frames = self.preprocess(frames)
            else:
                frames = self.not_norm_preprocess(frames)
                frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames

    def _get_obs(self, label, frame_ids, cam_id, pre_encode):
        if cam_id is None:
            temp_cam_id = random.choice(self.cam_ids)
        else:
            temp_cam_id = cam_id
        frames = self._get_frames(label, frame_ids, cam_id=temp_cam_id, pre_encode=pre_encode)
        return frames, temp_cam_id

    def _get_text(self, label):
        return label[self._text_key]

    def _get_actions(self, label, frame_ids):
        """Load actions directly from JSON annotation.
        
        AgiBotWorld actions are pre-computed 36-dim vectors already scaled to [0, 1].
        We return actions for frames 1 to sequence_length (not the first frame).
        """
        all_actions = np.array(label["action"])
        # Get actions corresponding to frame_ids[1:] (actions leading to those frames)
        action_frame_ids = frame_ids[1:]  # sequence_length - 1 actions
        actions = all_actions[action_frame_ids]
        
        assert actions.shape == (self.sequence_length - 1, self.action_dim), (
            f"Expected actions shape ({self.sequence_length - 1}, {self.action_dim}), "
            f"got {actions.shape}"
        )
        return torch.from_numpy(actions).float()

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
                actions = self._get_actions(label, frame_ids)
                data["action"] = actions

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
