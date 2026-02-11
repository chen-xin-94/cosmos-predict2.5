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

import json
import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.action.datasets.dataset_agibot import (
    ActionConditionedMultiViewDataset_AGIBOT,
)
from cosmos_predict2._src.predict2.action.datasets.dataset_droid import (
    ActionConditionedMultiViewDataset_DROID,
)
from cosmos_predict2._src.predict2.action.datasets.dataset_df import Dataset_3D_DF
from cosmos_predict2._src.predict2.action.datasets.dataset_local import Dataset_3D

try:
    from cosmos_predict2._src.predict2.action.configs.action_conditioned.experiment.gr00t_customized_gr1 import (
        register_gr00t_customized_gr1_data,
    )
except ImportError:
    register_gr00t_customized_gr1_data = None

# bridge dataset path
bridge_base_path = "datasets/opensource_robotdata/bridge"
bridge_train_annotation_path = os.path.join(bridge_base_path, "annotation/train")
bridge_val_annotation_path = os.path.join(bridge_base_path, "annotation/val")
bridge_test_annotation_path = os.path.join(bridge_base_path, "annotation/test")


# df dataset path
df_base_path = "datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand"
df_train_annotation_path = os.path.join(df_base_path, "annotation/train")
df_val_annotation_path = os.path.join(df_base_path, "annotation/val")
df_test_annotation_path = os.path.join(df_base_path, "annotation/test")

# agibot annotation path - will traverse all task subdirectories
agibot_base_annotation_path = "datasets/agibot/annotation"

# Smoke test dataset paths (smaller subsets for quick testing)
df_smoke_base_path = "/raid/chen.xin/repo/cosmos-predict2.5/assets/action_conditioned/concat_view/df"
df_smoke_train_annotation_path = os.path.join(df_smoke_base_path, "annotation/train")
df_smoke_val_annotation_path = os.path.join(df_smoke_base_path, "annotation/val")
df_smoke_test_annotation_path = os.path.join(df_smoke_base_path, "annotation/test")

agibot_smoke_base_annotation_path = "assets/action_conditioned/concat_view/agibot/annotation"

# droid dataset path
droid_base_path = "datasets/droid"
droid_train_annotation_path = os.path.join(droid_base_path, "annotation/train")
droid_val_annotation_path = os.path.join(droid_base_path, "annotation/val")
droid_test_annotation_path = os.path.join(droid_base_path, "annotation/test")
droid_action_stats_path = "assets/action_conditioned/concat_view/droid/stats.json"
droid_lerobot_meta_info_path = "/mnt/central_storage/data_pool/droid_lerobot/meta/info.json"

# droid smoke dataset path
droid_smoke_base_path = "datasets/smoke_test/droid"
droid_smoke_train_annotation_path = os.path.join(droid_smoke_base_path, "annotation/train")
droid_smoke_val_annotation_path = os.path.join(droid_smoke_base_path, "annotation/val")
droid_smoke_test_annotation_path = os.path.join(droid_smoke_base_path, "annotation/test")


def _has_json_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".json"):
                return True
    return False


def _resolve_val_annotation_path(preferred_path: str, fallback_path: str) -> str:
    # Some datasets are split as train/test only. Use test set for validation in that case.
    if _has_json_files(preferred_path):
        return preferred_path
    return fallback_path


def _load_droid_per_view_video_size(meta_info_path: str) -> list[int]:
    # Prefer dataset metadata for canonical resolution; fallback to known DROID default.
    default_size = [180, 320]
    if not os.path.isfile(meta_info_path):
        return default_size

    try:
        with open(meta_info_path, "r") as f:
            info = json.load(f)
        feature = info["features"]["observation.images.exterior_image_1_left"]
        shape = feature["shape"]
        if len(shape) >= 2:
            return [int(shape[0]), int(shape[1])]
    except Exception:
        return default_size

    return default_size


droid_effective_val_annotation_path = _resolve_val_annotation_path(
    droid_val_annotation_path,
    droid_test_annotation_path,
)
droid_smoke_effective_val_annotation_path = _resolve_val_annotation_path(
    droid_smoke_val_annotation_path,
    droid_smoke_test_annotation_path,
)
droid_per_view_video_size = _load_droid_per_view_video_size(droid_lerobot_meta_info_path)


# experiment for next-frame prediction
bridge_train_dataset = L(Dataset_3D)(
    train_annotation_path=bridge_train_annotation_path,
    val_annotation_path=bridge_val_annotation_path,
    test_annotation_path=bridge_test_annotation_path,
    video_path=bridge_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=1,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
)
bridge_val_dataset = L(Dataset_3D)(
    train_annotation_path=bridge_train_annotation_path,
    val_annotation_path=bridge_val_annotation_path,
    test_annotation_path=bridge_test_annotation_path,
    video_path=bridge_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=1,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="val",
)

# experiment for action-sequence video prediction
bridge_13frame_480_640_train_dataset = L(Dataset_3D)(
    train_annotation_path=bridge_train_annotation_path,
    val_annotation_path=bridge_val_annotation_path,
    test_annotation_path=bridge_test_annotation_path,
    video_path=bridge_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="train",
)
bridge_13frame_480_640_val_dataset = L(Dataset_3D)(
    train_annotation_path=bridge_train_annotation_path,
    val_annotation_path=bridge_val_annotation_path,
    test_annotation_path=bridge_test_annotation_path,
    video_path=bridge_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="val",
)

################### DF Dataset ###################
df_franka_single_arm_train_dataset = L(Dataset_3D_DF)(
    train_annotation_path=df_train_annotation_path,
    val_annotation_path=df_val_annotation_path,
    test_annotation_path=df_test_annotation_path,
    video_path=df_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=1,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="train",
)
df_franka_single_arm_val_dataset = L(Dataset_3D_DF)(
    train_annotation_path=df_train_annotation_path,
    val_annotation_path=df_val_annotation_path,
    test_annotation_path=df_test_annotation_path,
    video_path=df_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=1,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[256, 320],
    val_start_frame_interval=1,
    mode="val",
)

# experiment for action-sequence video prediction
df_franka_single_arm_13frame_480_640_train_dataset = L(Dataset_3D_DF)(
    train_annotation_path=df_train_annotation_path,
    val_annotation_path=df_val_annotation_path,
    test_annotation_path=df_test_annotation_path,
    video_path=df_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="train",
)
df_franka_single_arm_13frame_480_640_val_dataset = L(Dataset_3D_DF)(
    train_annotation_path=df_train_annotation_path,
    val_annotation_path=df_val_annotation_path,
    test_annotation_path=df_test_annotation_path,
    video_path=df_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0],
    accumulate_action=False,
    video_size=[480, 640],
    val_start_frame_interval=1,
    mode="val",
)
################################################

# ------------------------------------------------------------


# create dataloader for each dataset
def get_sampler(dataset):
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def build_webdataset(webdataset_instance, **kwargs):
    """Helper function to build WebDataset from a WebDataset instance.

    WebDatasets need to call build_dataset() to get the actual iterable dataset
    that can be used with DataLoader.

    Args:
        webdataset_instance: An instantiated WebDataset object.
        **kwargs: Additional parameters to override on the webdataset instance
            before building. This allows experiment configs to override parameters
            like gripper_rescale_factor, num_action_per_chunk, etc.
    """
    # Apply any parameter overrides to the webdataset instance
    for key, value in kwargs.items():
        if hasattr(webdataset_instance, key):
            setattr(webdataset_instance, key, value)
    return webdataset_instance.build_dataset()


bridge_train_dataloader = L(DataLoader)(
    dataset=bridge_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_train_dataset),
    batch_size=1,
    drop_last=True,
)
bridge_val_dataloader = L(DataLoader)(
    dataset=bridge_val_dataset,
    sampler=L(get_sampler)(dataset=bridge_val_dataset),
    batch_size=1,
    drop_last=True,
)

bridge_13frame_480_640_train_dataloader = L(DataLoader)(
    dataset=bridge_13frame_480_640_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_13frame_480_640_train_dataset),
    batch_size=1,
    drop_last=True,
)
bridge_13frame_480_640_val_dataloader = L(DataLoader)(
    dataset=bridge_13frame_480_640_val_dataset,
    sampler=L(get_sampler)(dataset=bridge_13frame_480_640_val_dataset),
    batch_size=1,
    drop_last=True,
)

################### DF Dataloader ###################
df_franka_single_arm_train_dataloader = L(DataLoader)(
    dataset=df_franka_single_arm_train_dataset,
    sampler=L(get_sampler)(dataset=bridge_train_dataset),
    batch_size=1,
    drop_last=True,
)
df_franka_single_arm_val_dataloader = L(DataLoader)(
    dataset=df_franka_single_arm_val_dataset,
    sampler=L(get_sampler)(dataset=bridge_val_dataset),
    batch_size=1,
    drop_last=True,
)

df_franka_single_arm_13frame_480_640_train_dataloader = L(DataLoader)(
    dataset=df_franka_single_arm_13frame_480_640_train_dataset,
    sampler=L(get_sampler)(dataset=df_franka_single_arm_13frame_480_640_train_dataset),
    batch_size=1,
    drop_last=True,
)
df_franka_single_arm_13frame_480_640_val_dataloader = L(DataLoader)(
    dataset=df_franka_single_arm_13frame_480_640_val_dataset,
    sampler=L(get_sampler)(dataset=df_franka_single_arm_13frame_480_640_val_dataset),
    batch_size=1,
    drop_last=True,
)
################################################


################### Multi-View DF Dataset (3 views, width-concatenated) ###################
# Import the multi-view dataset class
from cosmos_predict2._src.predict2.action.datasets.dataset_df import ActionConditionedMultiViewDataset_DF

# Multi-view 3-camera dataset: 448x1344 (3x448 width)
df_franka_multiview_13frame_448_1344_train_dataset = L(ActionConditionedMultiViewDataset_DF)(
    train_annotation_path=df_train_annotation_path,
    val_annotation_path=df_val_annotation_path,
    test_annotation_path=df_test_annotation_path,
    video_path=df_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],  # All 3 cameras: left, top, wrist
    accumulate_action=False,
    video_size=[448, 448],  # Native resolution per view, concat to 448x1344
    val_start_frame_interval=1,
    mode="train",
)
df_franka_multiview_13frame_448_1344_val_dataset = L(ActionConditionedMultiViewDataset_DF)(
    train_annotation_path=df_train_annotation_path,
    val_annotation_path=df_val_annotation_path,
    test_annotation_path=df_test_annotation_path,
    video_path=df_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],  # All 3 cameras: left, top, wrist
    accumulate_action=False,
    video_size=[448, 448],  # Native resolution per view, concat to 448x1344
    val_start_frame_interval=100,
    mode="val",
)

df_franka_multiview_13frame_448_1344_train_dataloader = L(DataLoader)(
    dataset=df_franka_multiview_13frame_448_1344_train_dataset,
    sampler=L(get_sampler)(dataset=df_franka_multiview_13frame_448_1344_train_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
df_franka_multiview_13frame_448_1344_val_dataloader = L(DataLoader)(
    dataset=df_franka_multiview_13frame_448_1344_val_dataset,
    sampler=L(get_sampler)(dataset=df_franka_multiview_13frame_448_1344_val_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
################################################


################### AgiBotWorld Multi-View Dataset (3 views, width-concatenated to 480x1920) ###################

# Multi-view 3-camera dataset: 480x1920 (3x640 width)
agibot_multiview_13frame_480_1920_train_dataset = L(ActionConditionedMultiViewDataset_AGIBOT)(
    base_annotation_path=agibot_base_annotation_path,
    video_path="",  # Not used, paths are absolute in JSON
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],  # All 3 cameras: hand_left, head, hand_right
    accumulate_action=False,
    video_size=[480, 640],  # Per-view resolution, concatenated to 480x1920
    val_start_frame_interval=1,
    mode="train",
)
agibot_multiview_13frame_480_1920_val_dataset = L(ActionConditionedMultiViewDataset_AGIBOT)(
    base_annotation_path=agibot_base_annotation_path,
    video_path="",  # Not used, paths are absolute in JSON
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],  # All 3 cameras: hand_left, head, hand_right
    accumulate_action=False,
    video_size=[480, 640],  # Per-view resolution, concatenated to 480x1920
    val_start_frame_interval=100,
    mode="val",
)

agibot_multiview_13frame_480_1920_train_dataloader = L(DataLoader)(
    dataset=agibot_multiview_13frame_480_1920_train_dataset,
    sampler=L(get_sampler)(dataset=agibot_multiview_13frame_480_1920_train_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
agibot_multiview_13frame_480_1920_val_dataloader = L(DataLoader)(
    dataset=agibot_multiview_13frame_480_1920_val_dataset,
    sampler=L(get_sampler)(dataset=agibot_multiview_13frame_480_1920_val_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
################################################


################### DROID Multi-View Dataset (3 views, width-concatenated to 180x960) ###################
droid_multiview_13frame_180_960_train_dataset = L(ActionConditionedMultiViewDataset_DROID)(
    train_annotation_path=droid_train_annotation_path,
    val_annotation_path=droid_effective_val_annotation_path,
    test_annotation_path=droid_test_annotation_path,
    video_path=droid_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],
    accumulate_action=False,
    video_size=droid_per_view_video_size,
    val_start_frame_interval=1,
    mode="train",
    action_stats_path=droid_action_stats_path,
    action_normalization="minmax",
)
droid_multiview_13frame_180_960_val_dataset = L(ActionConditionedMultiViewDataset_DROID)(
    train_annotation_path=droid_train_annotation_path,
    val_annotation_path=droid_effective_val_annotation_path,
    test_annotation_path=droid_test_annotation_path,
    video_path=droid_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],
    accumulate_action=False,
    video_size=droid_per_view_video_size,
    val_start_frame_interval=100,
    mode="val",
    action_stats_path=droid_action_stats_path,
    action_normalization="minmax",
)

droid_multiview_13frame_180_960_train_dataloader = L(DataLoader)(
    dataset=droid_multiview_13frame_180_960_train_dataset,
    sampler=L(get_sampler)(dataset=droid_multiview_13frame_180_960_train_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
droid_multiview_13frame_180_960_val_dataloader = L(DataLoader)(
    dataset=droid_multiview_13frame_180_960_val_dataset,
    sampler=L(get_sampler)(dataset=droid_multiview_13frame_180_960_val_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
################################################


################### Smoke Test DROID Multi-View Dataset (3 views, width-concatenated to 180x960) ###################
droid_multiview_13frame_180_960_smoke_train_dataset = L(ActionConditionedMultiViewDataset_DROID)(
    train_annotation_path=droid_smoke_train_annotation_path,
    val_annotation_path=droid_smoke_effective_val_annotation_path,
    test_annotation_path=droid_smoke_test_annotation_path,
    video_path=droid_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],
    accumulate_action=False,
    video_size=droid_per_view_video_size,
    val_start_frame_interval=1,
    mode="train",
    action_stats_path=droid_action_stats_path,
    action_normalization="minmax",
)
droid_multiview_13frame_180_960_smoke_val_dataset = L(ActionConditionedMultiViewDataset_DROID)(
    train_annotation_path=droid_smoke_train_annotation_path,
    val_annotation_path=droid_smoke_effective_val_annotation_path,
    test_annotation_path=droid_smoke_test_annotation_path,
    video_path=droid_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],
    accumulate_action=False,
    video_size=droid_per_view_video_size,
    val_start_frame_interval=100,
    mode="val",
    action_stats_path=droid_action_stats_path,
    action_normalization="minmax",
)

droid_multiview_13frame_180_960_smoke_train_dataloader = L(DataLoader)(
    dataset=droid_multiview_13frame_180_960_smoke_train_dataset,
    sampler=L(get_sampler)(dataset=droid_multiview_13frame_180_960_smoke_train_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
droid_multiview_13frame_180_960_smoke_val_dataloader = L(DataLoader)(
    dataset=droid_multiview_13frame_180_960_smoke_val_dataset,
    sampler=L(get_sampler)(dataset=droid_multiview_13frame_180_960_smoke_val_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
################################################


################### Smoke Test Multi-View DF Dataset (3 views, width-concatenated) ###################

# Smoke test multi-view 3-camera dataset: 448x1344 (3x448 width)
df_franka_multiview_13frame_448_1344_smoke_train_dataset = L(ActionConditionedMultiViewDataset_DF)(
    train_annotation_path=df_smoke_train_annotation_path,
    val_annotation_path=df_smoke_val_annotation_path,
    test_annotation_path=df_smoke_test_annotation_path,
    video_path=df_smoke_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],  # All 3 cameras: left, top, wrist
    accumulate_action=False,
    video_size=[448, 448],  # Native resolution per view, concat to 448x1344
    val_start_frame_interval=1,
    mode="train",
)
df_franka_multiview_13frame_448_1344_smoke_val_dataset = L(ActionConditionedMultiViewDataset_DF)(
    train_annotation_path=df_smoke_train_annotation_path,
    val_annotation_path=df_smoke_val_annotation_path,
    test_annotation_path=df_smoke_test_annotation_path,
    video_path=df_smoke_base_path,
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],  # All 3 cameras: left, top, wrist
    accumulate_action=False,
    video_size=[448, 448],  # Native resolution per view, concat to 448x1344
    val_start_frame_interval=100,
    mode="val",
)

df_franka_multiview_13frame_448_1344_smoke_train_dataloader = L(DataLoader)(
    dataset=df_franka_multiview_13frame_448_1344_smoke_train_dataset,
    sampler=L(get_sampler)(dataset=df_franka_multiview_13frame_448_1344_smoke_train_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
df_franka_multiview_13frame_448_1344_smoke_val_dataloader = L(DataLoader)(
    dataset=df_franka_multiview_13frame_448_1344_smoke_val_dataset,
    sampler=L(get_sampler)(dataset=df_franka_multiview_13frame_448_1344_smoke_val_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
################################################


################### Smoke Test AgiBotWorld Multi-View Dataset (3 views, width-concatenated to 480x1920) ###################

# Smoke test multi-view 3-camera dataset: 480x1920 (3x640 width)
agibot_multiview_13frame_480_1920_smoke_train_dataset = L(ActionConditionedMultiViewDataset_AGIBOT)(
    base_annotation_path=agibot_smoke_base_annotation_path,
    video_path="",  # Not used, paths are absolute in JSON
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],  # All 3 cameras: hand_left, head, hand_right
    accumulate_action=False,
    video_size=[480, 640],  # Per-view resolution, concatenated to 480x1920
    val_start_frame_interval=1,
    mode="train",
)
agibot_multiview_13frame_480_1920_smoke_val_dataset = L(ActionConditionedMultiViewDataset_AGIBOT)(
    base_annotation_path=agibot_smoke_base_annotation_path,
    video_path="",  # Not used, paths are absolute in JSON
    fps_downsample_ratio=1,
    num_action_per_chunk=12,
    cam_ids=[0, 1, 2],  # All 3 cameras: hand_left, head, hand_right
    accumulate_action=False,
    video_size=[480, 640],  # Per-view resolution, concatenated to 480x1920
    val_start_frame_interval=100,
    mode="val",
)

agibot_multiview_13frame_480_1920_smoke_train_dataloader = L(DataLoader)(
    dataset=agibot_multiview_13frame_480_1920_smoke_train_dataset,
    sampler=L(get_sampler)(dataset=agibot_multiview_13frame_480_1920_smoke_train_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
agibot_multiview_13frame_480_1920_smoke_val_dataloader = L(DataLoader)(
    dataset=agibot_multiview_13frame_480_1920_smoke_val_dataset,
    sampler=L(get_sampler)(dataset=agibot_multiview_13frame_480_1920_smoke_val_dataset),
    batch_size=1,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
)
################################################


def register_training_and_val_data():
    cs = ConfigStore.instance()
    from cosmos_predict2._src.predict2.configs.common.mock_data import MOCK_DATA_INTERLEAVE_CONFIG

    # Always register mock dataloaders to satisfy defaults when not overridden
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="mock",
        node=MOCK_DATA_INTERLEAVE_CONFIG,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="mock",
        node=MOCK_DATA_INTERLEAVE_CONFIG,
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="df_franka_single_arm_train",
        node=df_franka_single_arm_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="df_franka_single_arm_val",
        node=df_franka_single_arm_val_dataloader,
    )

    # 13 frame 480 640
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="df_franka_single_arm_13frame_480_640_train",
        node=df_franka_single_arm_13frame_480_640_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="df_franka_single_arm_13frame_480_640_val",
        node=df_franka_single_arm_13frame_480_640_val_dataloader,
    )

    # Multi-view 3-camera 13 frame 448x1344
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="df_franka_multiview_13frame_448_1344_train",
        node=df_franka_multiview_13frame_448_1344_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="df_franka_multiview_13frame_448_1344_val",
        node=df_franka_multiview_13frame_448_1344_val_dataloader,
    )

    # AgiBotWorld Multi-view 3-camera 13 frame 480x1920
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="agibot_multiview_13frame_480_1920_train",
        node=agibot_multiview_13frame_480_1920_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="agibot_multiview_13frame_480_1920_val",
        node=agibot_multiview_13frame_480_1920_val_dataloader,
    )

    # DROID Multi-view 3-camera 13 frame 180x960
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="droid_multiview_13frame_180_960_train",
        node=droid_multiview_13frame_180_960_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="droid_multiview_13frame_180_960_val",
        node=droid_multiview_13frame_180_960_val_dataloader,
    )

    # Smoke Test DROID Multi-view 3-camera 13 frame 180x960
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="droid_multiview_13frame_180_960_smoke_train",
        node=droid_multiview_13frame_180_960_smoke_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="droid_multiview_13frame_180_960_smoke_val",
        node=droid_multiview_13frame_180_960_smoke_val_dataloader,
    )

    # Smoke Test Multi-view 3-camera 13 frame 448x1344
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="df_franka_multiview_13frame_448_1344_smoke_train",
        node=df_franka_multiview_13frame_448_1344_smoke_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="df_franka_multiview_13frame_448_1344_smoke_val",
        node=df_franka_multiview_13frame_448_1344_smoke_val_dataloader,
    )

    # Smoke Test AgiBotWorld Multi-view 3-camera 13 frame 480x1920
    cs.store(
        group="data_train",
        package="dataloader_train",
        name="agibot_multiview_13frame_480_1920_smoke_train",
        node=agibot_multiview_13frame_480_1920_smoke_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="agibot_multiview_13frame_480_1920_smoke_val",
        node=agibot_multiview_13frame_480_1920_smoke_val_dataloader,
    )

    # Register gr00t_customized_gr1 data
    if register_gr00t_customized_gr1_data is not None:
        register_gr00t_customized_gr1_data()
