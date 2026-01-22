# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Configuration for action-conditioned multiview inference.

from pathlib import Path
from typing import List

from cosmos_predict2.config import (
    CommonInferenceArguments,
    CommonSetupArguments,
    Guidance,
    ModelKey,
    ModelVariant,
    get_model_literal,
    get_overrides_cls,
)

DEFAULT_MODEL_KEY = ModelKey(variant=ModelVariant.AUTO_MULTIVIEW)


class ActionMultiviewSetupArguments(CommonSetupArguments):
    """Setup arguments for action-conditioned multiview inference."""

    # Override defaults
    # pyrefly: ignore  # invalid-annotation
    model: get_model_literal([ModelVariant.AUTO_MULTIVIEW]) = DEFAULT_MODEL_KEY.name


class ActionMultiviewInferenceArguments(CommonInferenceArguments):
    """Inference arguments for action-conditioned multiview inference."""

    # Required parameters
    input_root: Path
    """Input root directory."""
    input_json_sub_folder: str
    """Input JSON sub-folder path."""

    # Output parameters
    save_root: Path = Path("results/action_multiview")
    """Save root directory."""

    # Model parameters
    chunk_size: int = 12
    """Number of actions per chunk."""
    chunk_overlap: int = 1
    """Frame overlap between chunks."""
    guidance: Guidance = 7
    """Guidance value."""
    resolution: str = "none"
    """Resolution of each view (H,W)."""
    num_steps: int = 35
    """Number of diffusion steps."""

    # Dataset-specific parameters
    camera_ids: List[int] = [0, 1, 2]
    """Camera IDs to use in order."""
    camera_keys: List[str] = ["frame_camera_left", "frame_camera_top", "wrist_camera"]
    """Camera keys corresponding to camera_ids."""
    start: int = 0
    """Start index for processing files."""
    end: int = 100
    """End index for processing files."""
    fps_downsample_ratio: int = 1
    """FPS downsample ratio."""
    gripper_scale: float = 1.0
    """Gripper scale factor."""
    gripper_key: str = "continuous_gripper_state"
    """Key for gripper state in JSON data."""
    state_key: str = "state"
    """Key for robot state in JSON data."""

    # Inference options
    start_frame_idx: int = 0
    """Start frame index."""
    save_fps: int = 20
    """FPS for saving output videos."""
    num_latent_conditional_frames: int = 1
    """Number of latent conditional frames (0, 1 or 2)."""
    use_text: bool = True
    """Whether to condition on text."""

    # Action processing parameters
    action_scaler: float = 20.0
    """Action scaling factor."""
    use_quat: bool = False
    """Whether to use quaternion representation for rotations."""


ActionMultiviewInferenceOverrides = get_overrides_cls(ActionMultiviewInferenceArguments, exclude=["name"])
