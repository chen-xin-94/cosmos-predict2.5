# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Action-conditioned multiview video generation inference.

import json
import os
from glob import glob

import mediapy
import numpy as np
import torch
from decord import VideoReader
from loguru import logger

from cosmos_predict2._src.imaginaire.utils import distributed
from cosmos_predict2._src.predict2.action.datasets.dataset_utils import quat2rotm, rotm2euler, rotm2quat
from cosmos_predict2._src.predict2_multiview.scripts.inference import NUM_CONDITIONAL_FRAMES_KEY, Vid2VidInference
from cosmos_predict2.action_multiview_config import (
    ActionMultiviewInferenceArguments,
    ActionMultiviewSetupArguments,
)
from cosmos_predict2.config import MODEL_CHECKPOINTS, VIDEO_EXTENSIONS


def _get_robot_states(label, state_key="state", gripper_key="continuous_gripper_state"):
    all_states = np.array(label[state_key])
    all_cont_gripper_states = np.array(label[gripper_key])
    return all_states, all_cont_gripper_states


def _get_actions(arm_states, gripper_states, sequence_length, use_quat=False):
    if use_quat:
        action = np.zeros((sequence_length - 1, 8))
    else:
        action = np.zeros((sequence_length - 1, 7))

    for k in range(1, sequence_length):
        prev_xyz = arm_states[k - 1, 0:3]
        prev_quat = arm_states[k - 1, 3:7]
        prev_rotm = quat2rotm(prev_quat)
        curr_xyz = arm_states[k, 0:3]
        curr_quat = arm_states[k, 3:7]
        curr_gripper = gripper_states[k]
        curr_rotm = quat2rotm(curr_quat)
        rel_xyz = np.dot(prev_rotm.T, curr_xyz - prev_xyz)
        rel_rotm = prev_rotm.T @ curr_rotm

        if use_quat:
            rel_rot = rotm2quat(rel_rotm)
            action[k - 1, 0:3] = rel_xyz
            action[k - 1, 3:7] = rel_rot
            action[k - 1, 7] = curr_gripper
        else:
            rel_rot = rotm2euler(rel_rotm)
            action[k - 1, 0:3] = rel_xyz
            action[k - 1, 3:6] = rel_rot
            action[k - 1, 6] = curr_gripper
    return action


def get_action_sequence_from_states(
    data,
    fps_downsample_ratio=1,
    use_quat=False,
    state_key="state",
    gripper_scale=1.0,
    gripper_key="continuous_gripper_state",
    action_scaler=20.0,
):
    arm_states, cont_gripper_states = _get_robot_states(data, state_key, gripper_key)
    actions = _get_actions(
        arm_states[::fps_downsample_ratio],
        cont_gripper_states[::fps_downsample_ratio],
        len(data[state_key][::fps_downsample_ratio]),
        use_quat=use_quat,
    )
    actions *= np.array(
        [action_scaler, action_scaler, action_scaler, action_scaler, action_scaler, action_scaler, gripper_scale]
    )
    return actions


def _read_conditional_frames(video_path: str, frame_ids: list[int], resolution: str) -> np.ndarray:
    vr = VideoReader(video_path)
    frames = vr.get_batch(frame_ids).asnumpy()
    if resolution != "none":
        try:
            h, w = map(int, resolution.split(","))
            frames = np.stack([mediapy.resize_image(frame, (h, w)) for frame in frames], axis=0)
        except Exception as e:
            logger.warning(f"Failed to resize image to {resolution}: {e}")
    return frames.astype(np.uint8)


def _build_multiview_input(frames_per_view: list[np.ndarray]) -> torch.Tensor:
    # frames_per_view: list of (T, H, W, C) uint8
    view_tensors = []
    for frames in frames_per_view:
        frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (T, C, H, W)
        view_tensors.append(frames_t)
    video = torch.cat(view_tensors, dim=0)  # (V*T, C, H, W)
    video = video.permute(1, 0, 2, 3)  # (C, V*T, H, W)
    return video.unsqueeze(0)  # (B=1, C, V*T, H, W)


def inference(
    setup_args: ActionMultiviewSetupArguments,
    inference_args: ActionMultiviewInferenceArguments,
):
    """Run action-conditioned multiview inference with temporal concatenation."""
    torch.enable_grad(False)

    if inference_args.num_latent_conditional_frames not in [0, 1, 2]:
        raise ValueError(
            "num_latent_conditional_frames must be 0, 1 or 2, "
            f"but got {inference_args.num_latent_conditional_frames}"
        )

    if inference_args.num_latent_conditional_frames > 1:
        has_videos = False
        for file_name in os.listdir(inference_args.input_root):
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in VIDEO_EXTENSIONS:
                has_videos = True
                break
        if not has_videos:
            raise ValueError(
                f"num_latent_conditional_frames={inference_args.num_latent_conditional_frames} > 1 requires videos, "
                f"but no videos found in {inference_args.input_root}"
            )

    checkpoint = MODEL_CHECKPOINTS[setup_args.model_key]
    experiment = setup_args.experiment or checkpoint.experiment
    # pyrefly: ignore  # missing-attribute
    checkpoint_path = setup_args.checkpoint_path or checkpoint.s3.uri
    if experiment is None:
        raise ValueError("Experiment name must be provided either in setup args or checkpoint metadata")

    pipe = Vid2VidInference(
        experiment_name=experiment,
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        context_parallel_size=setup_args.context_parallel_size,
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")

    input_video_path = inference_args.input_root
    input_json_path = inference_args.input_root / inference_args.input_json_sub_folder
    input_json_list = glob(str(input_json_path / "*.json"))

    rank0 = True
    # pyrefly: ignore  # unsupported-operation
    if setup_args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    inference_args.save_root.mkdir(parents=True, exist_ok=True)

    for annotation_path in input_json_list[inference_args.start : inference_args.end]:
        with open(annotation_path, "r") as f:
            json_data = json.load(f)

        video_paths = []
        for cam_id in inference_args.camera_ids:
            if isinstance(json_data["videos"][cam_id], dict):
                video_paths.append(str(input_video_path / json_data["videos"][cam_id]["video_path"]))
            else:
                video_paths.append(str(input_video_path / json_data["videos"][cam_id]))

        text = json_data["text"] if inference_args.use_text else ""

        actions = get_action_sequence_from_states(
            json_data,
            fps_downsample_ratio=inference_args.fps_downsample_ratio,
            state_key=inference_args.state_key,
            gripper_scale=inference_args.gripper_scale,
            gripper_key=inference_args.gripper_key,
            action_scaler=inference_args.action_scaler,
            use_quat=inference_args.use_quat,
        )

        img_name = annotation_path.split("/")[-1].split(".")[0]
        chunk_video_name = str(inference_args.save_root / f"{img_name}_chunk.mp4")
        if os.path.exists(chunk_video_name):
            logger.info(f"Video already exists: {chunk_video_name}")
            continue

        # Initial conditioning frames per view
        num_cond = inference_args.num_latent_conditional_frames
        frame_ids = list(range(inference_args.start_frame_idx, inference_args.start_frame_idx + max(num_cond, 1)))
        cond_frames_per_view = []
        for path in video_paths:
            frames = _read_conditional_frames(path, frame_ids, inference_args.resolution)
            if num_cond == 0:
                frames = np.zeros_like(frames[:1])
            cond_frames_per_view.append(frames[: max(num_cond, 1)])

        chunk_videos = []

        chunk_size_actions = inference_args.chunk_size
        chunk_stride_actions = inference_args.chunk_size + 1 - inference_args.chunk_overlap
        if chunk_stride_actions <= 0:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        for chunk_idx, start_action in enumerate(range(inference_args.start_frame_idx, len(actions), chunk_stride_actions)):
            actions_chunk = actions[start_action : start_action + chunk_size_actions]
            if actions_chunk.shape[0] != chunk_size_actions:
                pad_len = chunk_size_actions - actions_chunk.shape[0]
                if pad_len > 0:
                    pad_actions = np.zeros((pad_len, actions_chunk.shape[1]), dtype=actions_chunk.dtype)
                    actions_chunk = np.concatenate([actions_chunk, pad_actions], axis=0)

            num_video_frames = actions_chunk.shape[0] + 1
            frames_per_view = []
            for frames in cond_frames_per_view:
                if num_cond == 0:
                    cond_block = np.zeros((0,) + frames.shape[1:], dtype=frames.dtype)
                else:
                    cond_block = frames[-num_cond:]
                zeros = np.zeros(
                    (num_video_frames - cond_block.shape[0],) + cond_block.shape[1:], dtype=cond_block.dtype
                )
                frames_per_view.append(np.concatenate([cond_block, zeros], axis=0))

            vid_input = _build_multiview_input(frames_per_view)
            vid_input = vid_input.to(torch.uint8)

            data_batch = {
                "video": vid_input,
                "ai_caption": [[text for _ in inference_args.camera_ids]],
                "view_indices": torch.tensor(
                    [i for i in range(len(inference_args.camera_ids)) for _ in range(num_video_frames)],
                    dtype=torch.int64,
                ).unsqueeze(0),
                "view_indices_selection": torch.tensor(list(range(len(inference_args.camera_ids))), dtype=torch.int64).unsqueeze(
                    0
                ),
                "sample_n_views": torch.tensor(len(inference_args.camera_ids), dtype=torch.int64),
                "num_video_frames_per_view": torch.tensor(num_video_frames, dtype=torch.int64),
                "camera_keys_selection": [inference_args.camera_keys],
                "front_cam_view_idx_sample_position": torch.tensor(0, dtype=torch.int64),
                "ref_cam_view_idx_sample_position": torch.tensor(0, dtype=torch.int64),
                "fps": torch.tensor(float(inference_args.save_fps), dtype=torch.float32),
                "padding_mask": torch.zeros(1, vid_input.shape[-2], vid_input.shape[-1], dtype=torch.float32),
                "action": torch.from_numpy(actions_chunk).float().unsqueeze(0),
            }

            data_batch[NUM_CONDITIONAL_FRAMES_KEY] = num_cond if chunk_idx == 0 else inference_args.chunk_overlap

            video = pipe.generate_from_batch(
                data_batch,
                guidance=inference_args.guidance,
                seed=start_action,
                stack_mode="time",
                num_steps=inference_args.num_steps,
            )

            video_clamped = (
                (torch.clamp(video[0], 0, 1) * 255).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
            )  # (V*T, H, W, C)
            v = len(inference_args.camera_ids)
            video_views = video_clamped.reshape(v, num_video_frames, *video_clamped.shape[1:])

            # Update conditioning frames for next chunk (use last overlap frames)
            overlap = inference_args.chunk_overlap
            if overlap > 0:
                cond_frames_per_view = [video_views[i, -overlap:] for i in range(v)]
                num_cond = overlap

            if chunk_idx == 0:
                chunk_videos.append(video_views)
            else:
                chunk_videos.append(video_views[:, : num_video_frames - overlap])

        # Concatenate chunks per view
        per_view_out = []
        for view_idx in range(len(inference_args.camera_ids)):
            per_view_out.append(np.concatenate([chunk[view_idx] for chunk in chunk_videos], axis=0))

        final_video = np.concatenate(per_view_out, axis=0)

        if rank0:
            mediapy.write_video(chunk_video_name, final_video, fps=inference_args.save_fps)
            logger.info(f"Saved video to {chunk_video_name}")

    if setup_args.context_parallel_size > 1:
        torch.distributed.barrier()
    pipe.cleanup()
