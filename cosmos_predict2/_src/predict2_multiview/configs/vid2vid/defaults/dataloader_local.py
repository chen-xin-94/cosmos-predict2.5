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

"""Local file-based dataloader configurations."""

import torch.distributed as dist
from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_predict2._src.predict2_multiview.datasets.action_local import ActionMultiViewDatasetDF
from cosmos_predict2._src.predict2_multiview.datasets.local import WaymoLocalDataset
from cosmos_predict2._src.predict2_multiview.datasets.multiview import AugmentationConfig, collate_fn


def register_waymo_dataloader() -> None:
    """Register local file-based dataloader configurations."""

    cs = ConfigStore.instance()

    waymo_dataset = L(WaymoLocalDataset)(
        video_file_dirs=["datasets/multiview/waymo/input"],
        augmentation_config=L(AugmentationConfig)(
            resolution_hw=(720, 1280),
            fps_downsample_factor=1,
            num_video_frames=29,
            camera_keys=[
                "pinhole_front",
                "pinhole_front_right",
                "pinhole_side_right",
                "pinhole_side_left",
                "pinhole_front_left",
            ],
            camera_view_mapping={
                "pinhole_front": 0,
                "pinhole_front_right": 1,
                "pinhole_side_right": 2,
                # no pinhole_back in the dataset, so skip ID 3
                "pinhole_side_left": 4,
                "pinhole_front_left": 5,
                # no pinehole_front_tele in the dataset skip ID 6
            },
            camera_video_key_mapping={
                "pinhole_front": "video_pinhole_front",
                "pinhole_front_right": "video_pinhole_front_right",
                "pinhole_side_right": "video_pinhole_side_right",
                "pinhole_side_left": "video_pinhole_side_left",
                "pinhole_front_left": "video_pinhole_front_left",
            },
            camera_caption_key_mapping={
                "pinhole_front": "caption_pinhole_front",
                "pinhole_front_right": "caption_pinhole_front_right",
                "pinhole_side_right": "caption_pinhole_side_right",
                "pinhole_side_left": "caption_pinhole_side_left",
                "pinhole_front_left": "caption_pinhole_front_left",
            },
            caption_probability={
                "long": 1.0,
            },
            single_caption_camera_name="pinhole_front",
            add_view_prefix_to_caption=True,
            camera_prefix_mapping={
                "pinhole_front": "The video is captured from a camera mounted on a car. The camera is facing forward.",
                "pinhole_front_right": "The video is captured from a camera mounted on a car. The camera is facing to the front right.",
                "pinhole_side_right": "The video is captured from a camera mounted on a car. The camera is facing to the side right.",
                "pinhole_side_left": "The video is captured from a camera mounted on a car. The camera is facing to the side left.",
                "pinhole_front_left": "The video is captured from a camera mounted on a car. The camera is facing to the front left.",
            },
        ),
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name=f"waymo",
        node=L(get_generic_dataloader)(
            dataset=waymo_dataset,
            sampler=L(get_sampler)(dataset=waymo_dataset) if dist.is_initialized() else None,
            collate_fn=collate_fn,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )

    cs.store(
        group="data_val",
        package="dataloader_val",
        name=f"waymo",
        node=L(get_generic_dataloader)(
            dataset=waymo_dataset,
            sampler=L(get_sampler)(dataset=waymo_dataset) if dist.is_initialized() else None,
            collate_fn=collate_fn,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )


def register_awam_dataloader() -> None:
    """Register local file-based AWAM dataloader configurations."""

    cs = ConfigStore.instance()

    # Action multiview local dataset (AVLA-style JSONs with 3 views)
    base_path = "datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand"
    # base_path = "datasets_example/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand"

    annotation_path = f"{base_path}/annotation"
    avla_action_multiview_dataset = L(ActionMultiViewDatasetDF)(
        train_annotation_path=f"{annotation_path}/train",
        val_annotation_path=f"{annotation_path}/val",
        test_annotation_path=f"{annotation_path}/test",
        video_path="",  # Absolute paths in JSON
        fps_downsample_ratio=1,
        num_action_per_chunk=12,
        cam_ids=[0, 1, 2],
        camera_keys=["frame_camera_left", "frame_camera_top", "wrist_camera"],
        accumulate_action=False,
        video_size=[448, 448],
        val_start_frame_interval=100,
        mode="train",
        state_key="state",
        gripper_key="continuous_gripper_state",
        text_key="text",
    )
    avla_action_multiview_val_dataset = L(ActionMultiViewDatasetDF)(
        train_annotation_path=f"{annotation_path}/train",
        val_annotation_path=f"{annotation_path}/val",
        test_annotation_path=f"{annotation_path}/test",
        video_path="",
        fps_downsample_ratio=1,
        num_action_per_chunk=12,
        cam_ids=[0, 1, 2],
        camera_keys=["frame_camera_left", "frame_camera_top", "wrist_camera"],
        accumulate_action=False,
        video_size=[448, 448],
        val_start_frame_interval=100,
        mode="val",
        state_key="state",
        gripper_key="continuous_gripper_state",
        text_key="text",
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="avla_action_multiview_13frame_448_448_train",
        node=L(get_generic_dataloader)(
            dataset=avla_action_multiview_dataset,
            sampler=L(get_sampler)(dataset=avla_action_multiview_dataset) if dist.is_initialized() else None,
            collate_fn=collate_fn,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="avla_action_multiview_13frame_448_448_val",
        node=L(get_generic_dataloader)(
            dataset=avla_action_multiview_val_dataset,
            sampler=L(get_sampler)(dataset=avla_action_multiview_val_dataset) if dist.is_initialized() else None,
            collate_fn=collate_fn,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )
