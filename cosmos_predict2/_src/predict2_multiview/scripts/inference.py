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
This script is based on projects/cosmos/diffusion/v2/inference/vid2vid.py

To run inference on the training data (as visualization/debugging), use:
```bash
EXP=buttercup_predict2p5_2b_7views_res720p_fps30_t8_joint_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform_dropoutt0
ckpt_path=s3://bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_7views_res720p_fps30_t8_joint_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform_dropoutt0-0/checkpoints/iter_000012000/
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m cosmos_predict2._src.predict2_multiview.scripts.inference --experiment ${EXP} --ckpt_path ${ckpt_path} --context_parallel_size 8 --input_is_train_data --max_samples 1 --num_conditional_frames 0 --guidance 3 --save_root results/predict2_multiview/

EXP=predict2p5_2b_mv_7train7_res480p_fps15_t24_alpamayo_only_allcaption_uniform_nofps
ckpt_path=s3://bucket/cosmos_predict2_multiview/cosmos2p5_mv/predict2p5_2b_mv_7train7_res480p_fps15_t24_alpamayo_only_allcaption_uniform_nofps-0/checkpoints/iter_000020000/
PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 -m cosmos_predict2._src.predict2_multiview.scripts.inference --experiment ${EXP} --ckpt_path ${ckpt_path} --context_parallel_size 1 --input_is_train_data --max_samples 1 --num_conditional_frames 0 --guidance 3 --save_root results/predict2_multiview_480p_20k/
```
"""

import argparse
import os

import torch
import torch as th
from einops import rearrange
from megatron.core import parallel_state

from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import distributed, log
from cosmos_predict2._src.imaginaire.visualize.video import save_img_or_video
from cosmos_predict2._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_predict2._src.predict2_multiview.scripts.mv_visualize_helper import arrange_video_visualization

NUM_CONDITIONAL_FRAMES_KEY = "num_conditional_frames"


def to_model_input(data_batch, model):
    """
    Similar to misc.to, but avoid converting uint8 "video" to float
    """
    for k, v in data_batch.items():
        _v = v
        if isinstance(v, th.Tensor):
            _v = _v.cuda()
            if th.is_floating_point(v):
                _v = _v.to(**model.tensor_kwargs)
        data_batch[k] = _v
    return data_batch


class Vid2VidInference:
    """
    Handles the Vid2Vid inference process, including model loading, data preparation,
    and video generation from an image/video and text prompt. Now supports context parallelism.
    """

    def __init__(
        self,
        experiment_name: str,
        ckpt_path: str,
        s3_credential_path: str = "",
        context_parallel_size: int = 1,
        experiment_opts: list[str] = [],
    ):
        """
        Initializes the Vid2VidInference class.

        Loads the diffusion model and its configuration based on the provided
        experiment name and checkpoint path. Sets up distributed processing if needed.

        Args:
            experiment_name (str): Name of the experiment configuration.
            ckpt_path (str): Path to the model checkpoint (local or S3).
            s3_credential_path (str): Path to S3 credentials file (if loading from S3).
            context_parallel_size (int): Number of GPUs for context parallelism.
        """
        self.experiment_name = experiment_name
        self.ckpt_path = ckpt_path
        self.s3_credential_path = s3_credential_path
        self.context_parallel_size = context_parallel_size
        self.process_group = None
        self.experiment_opts = experiment_opts

        if "RANK" in os.environ:
            self._init_distributed()

        # Load the model and config
        model, config = load_model_from_checkpoint(
            experiment_name=self.experiment_name,
            s3_checkpoint_dir=self.ckpt_path,
            config_file="cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py",
            load_ema_to_reg=True,
            experiment_opts=self.experiment_opts,
        )

        # Enable context parallel on the model if using context parallelism
        self.rank0 = True
        if self.context_parallel_size > 1:
            model.net.enable_context_parallel(self.process_group)
            self.rank0 = distributed.get_rank() == 0

        self.model = model
        self.config = config
        self.batch_size = 1
        self.neg_t5_embeddings = None

    def _init_distributed(self):
        """Initialize distributed processing for context parallelism."""

        # Initialize distributed environment
        distributed.init()

        # Initialize model parallel states
        parallel_state.initialize_model_parallel(
            context_parallel_size=self.context_parallel_size,
        )

        # Get the process group for context parallel
        self.process_group = parallel_state.get_context_parallel_group()

        log.info(f"Initialized context parallel with size {self.context_parallel_size}")
        log.info(f"Current rank: {distributed.get_rank()}, World size: {distributed.get_world_size()}")

    def generate_from_batch(
        self,
        data_batch,
        guidance: int = 7,
        seed: int = 1,
        num_steps: int = 35,
        stack_mode: str = "time",
        use_negative_prompt: bool = True,
    ):
        """Generate video tensor from batch.

        Returns:
            Tensor with values in the range [0, 1]
            If stack mode is "time", the tensor is of shape (1, 3, v * t, h, w)
            If stack mode is "height", the tensor is of shape (1, 3, t, v * h, w)
            If stack mode is "width", the tensor is of shape (1, 3, t, h, v * w)
            If stack mode is "grid", the tensor is of shape (1, 3, t, 3 * h, 3 * w)
        """
        data_batch = to_model_input(data_batch, self.model)
        if self.model.config.text_encoder_config is not None and self.model.config.text_encoder_config.compute_online:
            self.model.inplace_compute_text_embeddings_online(data_batch)
        raw_data, x0, condition = self.model.get_data_and_condition(data_batch)
        sample = self.model.generate_samples_from_batch(
            data_batch,
            guidance=guidance,
            # make sure no mismatch and also works for cp
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            seed=seed,  # Fixed seed for reproducibility
            num_steps=num_steps,
            is_negative_prompt=use_negative_prompt,
        )
        # (bsz = 1, c = 3, t = n_camera * t, h, w)
        video = ((self.model.decode(sample) + 1.0) / 2.0).clamp(0, 1)

        # Arrange video according to stack_mode
        video = arrange_video_visualization(video, data_batch, method=stack_mode)
        return video

    def generate_from_batch_autoregressive(
        self,
        data_batch,
        num_chunks=2,
        chunk_overlap=2,
        guidance: int = 7,
        seed: int = 1,
        num_steps: int = 35,
        stack_mode: str = "time",
        use_negative_prompt: bool = True,
    ):
        """Generate video tensor from batch, with autoregressive mode enabled
        num_chunks: total number of single generation
        chunk_overlap: overlap the
        """
        data_batch = to_model_input(data_batch, self.model)
        if self.model.config.text_encoder_config is not None and self.model.config.text_encoder_config.compute_online:
            self.model.inplace_compute_text_embeddings_online(data_batch)

        n_views = len(data_batch["camera_keys_selection"][0])
        num_video_frames_per_view = data_batch["num_video_frames_per_view"][0]

        generated_chunks = []

        for i in range(num_chunks):
            log.info(f"start generate chunk {i + 1} / {num_chunks}")
            _, x0, _ = self.model.get_data_and_condition(data_batch)
            sample = self.model.generate_samples_from_batch(
                data_batch,
                guidance=guidance,
                # make sure no mismatch and also works for cp
                state_shape=x0.shape[1:],
                n_sample=x0.shape[0],
                seed=seed,  # Fixed seed for reproducibility
                num_steps=num_steps,
                is_negative_prompt=use_negative_prompt,
            )
            # (bsz = 1, c = 3, t = n_camera * t, h, w)
            decoded = self.model.decode(sample)
            chunk_video = ((decoded + 1.0) / 2.0).clamp(0, 1)[0]
            chunk_video = rearrange(chunk_video, "C (V T) H W -> V C T H W", V=n_views)
            if i == 0:
                generated_chunks.append(chunk_video)
            else:
                generated_chunks.append(chunk_video[:, :, chunk_overlap:])
            data_batch["num_conditional_frames"] = chunk_overlap
            data_batch["video"].zero_()
            for v in range(n_views):
                start_idx = num_video_frames_per_view * v
                overlaps = (
                    chunk_video[v, :, num_video_frames_per_view - chunk_overlap : num_video_frames_per_view] * 255
                )
                overlaps = overlaps.to(th.uint8).clamp(0, 255)
                data_batch["video"][:, :, start_idx : start_idx + chunk_overlap] = overlaps

        video = th.cat(generated_chunks, dim=2)
        video = rearrange(video, "V C T H W -> C (V T) H W", V=n_views).unsqueeze(0)

        # Arrange video according to stack_mode
        video = arrange_video_visualization(video, data_batch, method=stack_mode)
        return video

    def generate_autoregressive_from_batch(
        self,
        data_batch,
        num_output_frames: int,
        chunk_size: int,
        chunk_overlap: int,
        guidance: int = 7,
        seed: int = 1,
        num_conditional_frames: int = 1,
        num_steps: int = 35,
        stack_mode: str = "time",
        use_negative_prompt: bool = True,
    ) -> th.Tensor:
        """Generate multiview video using autoregressive sliding window over time-per-view.

        Returns:
            Tensor with values in the range [0, 1] and shape (B, C, V*T, H, W) when stack_mode="time".
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        # Model-required frames per view
        model_required_frames = self.model.tokenizer.get_pixel_num_frames(self.model.config.state_t)

        # Determine number of views
        sample_n_views = data_batch.get("sample_n_views", None)
        if isinstance(sample_n_views, th.Tensor):
            n_views = int(sample_n_views.item())
        else:
            n_views = int(sample_n_views) if sample_n_views is not None else 1

        # Initialize full-length input video (B, C, V, T, H, W) on same device as input
        video_input = data_batch["video"]
        device = video_input.device
        B, C, VT, H, W = video_input.shape
        num_video_frames_per_view = VT // n_views
        video_input_V = rearrange(video_input, "B C (V T) H W -> B C V T H W", V=n_views)

        full_input_video = th.zeros(
            B,
            C,
            n_views,
            num_output_frames,
            H,
            W,
            dtype=video_input.dtype,
            device=device,
        )
        if num_conditional_frames > 0:
            full_input_video[:, :, :, :num_conditional_frames] = video_input_V[:, :, :, :num_conditional_frames]

        # Calculate number of chunks
        effective_chunk_size = chunk_size - chunk_overlap
        remaining_after_first = num_output_frames - chunk_size
        if remaining_after_first <= 0:
            num_chunks = 1
        else:
            num_chunks = 1 + (remaining_after_first + effective_chunk_size - 1) // effective_chunk_size

        log.info(
            f"Generating {num_chunks} chunks with chunk_size={chunk_size}, chunk_overlap={chunk_overlap} "
            f"for {num_output_frames} total frames per view"
        )

        generated_chunks = []
        current_input_video = full_input_video.clone()

        for chunk_idx in range(num_chunks):
            start_frame = chunk_idx * effective_chunk_size
            end_frame = min(start_frame + chunk_size, num_output_frames)
            actual_chunk_size = end_frame - start_frame
            if start_frame >= num_output_frames:
                break

            # Extract chunk input and pad to model_required_frames
            chunk_input = current_input_video[:, :, :, start_frame:end_frame]  # (B, C, V, T, H, W)
            if actual_chunk_size < model_required_frames:
                padding_frames = model_required_frames - actual_chunk_size
                padding = th.zeros(
                    B,
                    C,
                    n_views,
                    padding_frames,
                    H,
                    W,
                    dtype=chunk_input.dtype,
                    device=device,
                )
                chunk_input = torch.cat([chunk_input, padding], dim=3)

            chunk_input = rearrange(chunk_input, "B C V T H W -> B C (V T) H W", V=n_views)

            # Determine num_conditional_frames for this chunk
            chunk_num_conditional = num_conditional_frames if chunk_idx == 0 else chunk_overlap

            # Build chunk batch
            chunk_batch = dict(data_batch)
            chunk_batch["video"] = chunk_input
            chunk_batch["num_video_frames_per_view"] = th.tensor(model_required_frames, dtype=th.int64)
            chunk_batch["view_indices"] = th.tensor(
                [i for i in range(n_views) for _ in range(model_required_frames)], dtype=th.int64
            ).unsqueeze(0)
            chunk_batch[NUM_CONDITIONAL_FRAMES_KEY] = chunk_num_conditional

            # Generate chunk
            chunk_video = self.generate_from_batch(
                chunk_batch,
                guidance=guidance,
                seed=seed + chunk_idx,
                num_steps=num_steps,
                stack_mode=stack_mode,
                use_negative_prompt=use_negative_prompt,
            )
            chunk_video = chunk_video.to(device)
            # chunk_video: (B, C, V*T, H, W) in [0,1]
            chunk_video = rearrange(chunk_video, "B C (V T) H W -> B C V T H W", V=n_views)
            chunk_video = chunk_video[:, :, :, :actual_chunk_size]

            if chunk_idx == 0:
                generated_chunks.append(chunk_video)
            else:
                generated_chunks.append(chunk_video[:, :, :, chunk_overlap:])

            if chunk_idx < num_chunks - 1:
                update_start = start_frame + chunk_num_conditional
                update_end = end_frame
                current_input_video[:, :, :, update_start:update_end] = chunk_video[:, :, :, chunk_num_conditional:]

        # Concatenate all chunks along time dimension per view
        final_video = torch.cat(generated_chunks, dim=3)
        final_video = rearrange(final_video, "B C V T H W -> B C (V T) H W", V=n_views)
        log.info(f"Generated final video with shape {final_video.shape}")
        return final_video

    def cleanup(self):
        """Clean up distributed resources."""
        if "RANK" in os.environ:
            import torch.distributed as dist
            from megatron.core import parallel_state

            if parallel_state.is_initialized():
                parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the Vid2Vid inference script."""
    parser = argparse.ArgumentParser(description="Image2World/Video2World inference script")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment config")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
        help="Path to the checkpoint. If not provided, will use the one specify in the config",
    )
    parser.add_argument("--s3_cred", type=str, default="credentials/s3_checkpoint.secret")
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Context parallel size (number of GPUs to split context over). Set to 8 for 8 GPUs",
    )
    # generation
    parser.add_argument("--guidance", type=int, default=7, help="Guidance value")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument("--seed", type=int, default=1, help="Guidance value")
    parser.add_argument("--num_conditional_frames", type=int, default=1, help="Number of conditional frames")
    # input
    parser.add_argument(
        "--input_is_train_data",
        action="store_true",
        help="Inference on the training data, the input_root will be ignored if this is set",
    )
    parser.add_argument("--run_mads_verification", action="store_true", help="Run MADS verification")
    parser.add_argument(
        "--mads_verification_prompt",
        type=str,
        default='The video opens with a view from inside a vehicle, positioned at an intersection under a clear blue sky. The camera angle is from the dashboard, offering a first-person perspective of the road ahead. The intersection is marked by multiple traffic lights and street signs, including one that reads "E Garden Blvd." A white van with "TM Stuckateur" branding is seen driving through the intersection, heading in the same direction as the viewer\'s vehicle. Other cars are also present, moving smoothly along the multi-lane road. As the vehicle starts to move forward, the camera pans slightly to the right, revealing more of the surroundings. The road is lined with trees on both sides, providing a natural canopy that filters the sunlight. The trees are lush and green, indicating it might be spring or summer. On the left side of the road, there is a large building with a sign that reads "GROCERY OUTLET," suggesting the presence of a retail store nearby. Further down the road, additional buildings and residential structures can be seen, hinting at a suburban or semi-urban area. The sun is bright and high in the sky, casting long shadows across the road. The light creates a warm, inviting atmosphere, enhancing the clarity of the scene. The road itself is well-maintained, with clear lane markings and directional arrows painted on the asphalt. Overhead, power lines run parallel to the road, supported by poles that also hold traffic lights and street lamps. As the vehicle continues its journey, the camera maintains a steady focus on the road ahead, capturing the smooth flow of traffic and the serene environment. The absence of heavy traffic or congestion adds to the tranquil mood of the scene. The overall ambiance is one of calm and order, with the interplay of natural and man-made elements creating a harmonious urban landscape. The gentle curve of the road and the soft glow of the setting sun add a sense of peacefulness to the drive, making the viewer feel as though they are part of this quiet, picturesque neighborhood.',
    )
    parser.add_argument(
        "--stack_mode",
        type=str,
        default="time",
        choices=["height", "width", "time", "grid"],
        help="Video stacking mode for visualization. grid will create a 3x3 grid of views.",
    )
    parser.add_argument("--input_root", type=str, default="assets/image2world", help="Input root")
    parser.add_argument("--save_root", type=str, default="results/image2world", help="Save root")
    parser.add_argument("--max_samples", type=int, default=20, help="Maximum number of samples to generate")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ["NVTE_FUSED_ATTN"] = "0"
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    th.enable_grad(False)
    args = parse_arguments()
    # Initialize the inference handler with context parallel support
    vid2vid_cli = Vid2VidInference(
        args.experiment, args.ckpt_path, args.s3_cred, context_parallel_size=args.context_parallel_size
    )
    mem_bytes = th.cuda.memory_allocated(device=th.device("cuda" if th.cuda.is_available() else "cpu"))
    log.info(f"GPU memory usage after model dcp.load: {mem_bytes / (1024**3):.2f} GB")

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    if args.context_parallel_size > 1:
        rank0 = distributed.get_rank() == 0

    os.makedirs(args.save_root, exist_ok=True)
    if args.input_is_train_data:
        dataloader = instantiate(vid2vid_cli.config.dataloader_train)
        for i, batch in enumerate(dataloader):
            if i >= args.max_samples:
                break
            if args.run_mads_verification:
                assert args.num_conditional_frames == 0, "MADS verification only supports 0 conditional frame"
                log.warning(f"Running MADS verification with prompt: {args.mads_verification_prompt[0:100]}...")
                batch["ai_caption"] = [args.mads_verification_prompt]
            batch[NUM_CONDITIONAL_FRAMES_KEY] = args.num_conditional_frames
            video = vid2vid_cli.generate_from_batch(
                batch, guidance=args.guidance, seed=args.seed, stack_mode=args.stack_mode
            )
            if rank0:
                save_name = f"mads_verification_{i}" if args.run_mads_verification else f"infer_from_train_{i}"
                save_img_or_video(video[0], f"{args.save_root}/{save_name}", fps=args.fps)
            if args.run_mads_verification:
                break
    else:
        raise NotImplementedError("Not implemented")
