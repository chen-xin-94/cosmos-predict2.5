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

from datetime import datetime

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path
from cosmos_predict2.config import MODEL_CHECKPOINTS, ModelKey

# Use the post-trained checkpoint which has the correct experiment reference
DEFAULT_CHECKPOINT = MODEL_CHECKPOINTS[ModelKey()]  # This uses post_trained=True by default


"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320
"""
ac_reason_embeddings_rectified_flow_2b_256_320 = LazyDict(
    dict(
        defaults=[
            DEFAULT_CHECKPOINT.experiment,
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {"override /data_train": "bridge_13frame_480_640_train"},
            {"override /data_val": "bridge_13frame_480_640_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="cosmos_predict_v2p5",
            name="2b_bridge_action_conditioned",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=2_000,
            # pyrefly: ignore  # missing-attribute
            load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        trainer=dict(
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=5000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                    save_s3=False,
                ),
                every_n_sample_ema=dict(
                    every_n=5000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                    save_s3=False,
                ),
                heart_beat=dict(
                    save_s3=False,
                ),
                iter_speed=dict(
                    hit_thres=100,
                    save_s3=False,
                ),
                device_monitor=dict(
                    save_s3=False,
                ),
                wandb=dict(
                    save_s3=False,
                ),
                wandb_10x=dict(
                    save_s3=False,
                ),
                dataloader_speed=dict(
                    save_s3=False,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        model=dict(
            config=dict(
                # NOTE: this should be 1 for the action conditioned model
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                # overwrite the probs to disable random num of conditional frames
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,
                net=dict(
                    action_dim=7,
                    num_action_per_chunk=12,
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=2,
            sampler=dict(
                dataset=dict(fps_downsample_ratio=1, video_size=[256, 320]),
            ),
            dataset=dict(fps_downsample_ratio=1, video_size=[256, 320]),
        ),
    ),
    flags={"allow_objects": True},
)

"""
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_df ~dataloader_train.dataloaders
"""
ac_reason_embeddings_rectified_flow_2b_256_320_df = LazyDict(
    dict(
        defaults=[
            "ac_reason_embeddings_rectified_flow_2b_256_320",  # inherit
            {"override /data_train": "df_franka_single_arm_13frame_480_640_train"},
            {"override /data_val": "df_franka_single_arm_13frame_480_640_val"},
            "_self_",
        ],
        job=dict(
            name="2b_df_action_wo_text_conditioned_10k_bs4_debugging",
        ),
        trainer=dict(
            max_iter=1_000,
        ),
        model=dict(
            config=dict(
                conditioner=dict(
                    text=dict(use_prompt=False),
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=4,
            sampler=dict(
                dataset=dict(fps_downsample_ratio=6, video_size=[256, 320]),
            ),
            dataset=dict(fps_downsample_ratio=6, video_size=[256, 320]),
        ),
    ),
    flags={"allow_objects": True},
)


"""
Multi-view 3-camera action-conditioned training
Resolution: 448x1344 (3 views of 448x448 concatenated along width)

torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=ac_reason_embeddings_rectified_flow_2b_multiview_448_1344 ~dataloader_train.dataloaders
"""
ac_reason_embeddings_rectified_flow_2b_multiview_448_1344 = LazyDict(
    dict(
        defaults=[
            DEFAULT_CHECKPOINT.experiment,  # Use the checkpoint's experiment directly
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {"override /data_train": "df_franka_multiview_13frame_448_1344_train"},
            {"override /data_val": "df_franka_multiview_13frame_448_1344_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="cosmos_predict_v2p5",
            name=f"2b_df_multiview_action_conditioned_448_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            wandb_mode="online",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=50_000,
            logging_iter=10,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(every_n=1_000, do_x0_prediction=False, guidance=[0, 3, 7], fps=16, save_s3=False),
                every_n_sample_ema=dict(every_n=1_000, do_x0_prediction=False, guidance=[0, 3, 7], fps=16, save_s3=False),
                heart_beat=dict(save_s3=False),
                iter_speed=dict(hit_thres=5, every_n=1, save_s3=False),
                device_monitor=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,
                net=dict(action_dim=7, num_action_per_chunk=12),
                conditioner=dict(text=dict(use_prompt=True)),  # Enable text conditioning
            ),
        ),
        dataloader_train=dict(
            batch_size=2,
            sampler=dict(dataset=dict(fps_downsample_ratio=6, video_size=[448, 448])),
            dataset=dict(fps_downsample_ratio=6, video_size=[448, 448]),
        ),
    ),
    flags={"allow_objects": True},
)


"""
AgiBotWorld Multi-view 3-camera action-conditioned training
Resolution: 480x1920 (3 views of 480x640 concatenated along width)
Action dimension: 36 (pre-scaled to [0, 1])

torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=ac_reason_embeddings_rectified_flow_2b_agibot_480_1920 ~dataloader_train.dataloaders
"""
ac_reason_embeddings_rectified_flow_2b_agibot_480_1920 = LazyDict(
    dict(
        defaults=[
            DEFAULT_CHECKPOINT.experiment,  # Use the checkpoint's experiment directly
            {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
            {"override /net": "cosmos_v1_2B_action_conditioned"},
            {"override /conditioner": "action_conditioned_video_conditioner"},
            {"override /data_train": "agibot_multiview_13frame_480_1920_train"},
            {"override /data_val": "agibot_multiview_13frame_480_1920_val"},
            "_self_",
        ],
        job=dict(
            project="cosmos_predict2_action_conditioned",
            group="cosmos_predict_v2p5",
            name=f"2b_agibot_multiview_action_conditioned_480_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            wandb_mode="online",
        ),
        optimizer=dict(
            lr=2 ** (-14.5),
            weight_decay=0.1,
        ),
        checkpoint=dict(
            save_iter=2_000,
            load_path=get_checkpoint_path(DEFAULT_CHECKPOINT.s3.uri),
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
        ),
        trainer=dict(
            max_iter=50_000,
            logging_iter=10,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=dict(every_n=1_000, do_x0_prediction=False, guidance=[0, 3, 7], fps=16, save_s3=False),
                every_n_sample_ema=dict(every_n=1_000, do_x0_prediction=False, guidance=[0, 3, 7], fps=16, save_s3=False),
                heart_beat=dict(save_s3=False),
                iter_speed=dict(hit_thres=5, every_n=1, save_s3=False),
                device_monitor=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(context_parallel_size=1),
        model=dict(
            config=dict(
                min_num_conditional_frames=1,
                max_num_conditional_frames=1,
                conditional_frames_probs=None,
                state_t=1 + 12 // 4,
                net=dict(action_dim=36, num_action_per_chunk=12),  # 36-dim actions for AgiBotWorld
                conditioner=dict(text=dict(use_prompt=True)),  # Enable text conditioning
            ),
        ),
        dataloader_train=dict(
            batch_size=2,
            sampler=dict(dataset=dict(fps_downsample_ratio=6, video_size=[480, 640])),
            dataset=dict(fps_downsample_ratio=6, video_size=[480, 640]),
        ),
    ),
    flags={"allow_objects": True},
)



cs = ConfigStore.instance()


"""
Smoke Test for Multi-view 3-camera action-conditioned training
Resolution: 448x1344 (3 views of 448x448 concatenated along width)
Uses smaller dataset from assets/action_conditioned/concat_view/df

torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=ac_reason_embeddings_rectified_flow_2b_multiview_448_1344_smoke ~dataloader_train.dataloaders
"""
ac_reason_embeddings_rectified_flow_2b_multiview_448_1344_smoke = LazyDict(
    dict(
        defaults=[
            "ac_reason_embeddings_rectified_flow_2b_multiview_448_1344",  # Inherit from the full experiment
            {"override /data_train": "df_franka_multiview_13frame_448_1344_smoke_train"},
            {"override /data_val": "df_franka_multiview_13frame_448_1344_smoke_val"},
            "_self_",
        ],
        job=dict(
            name=f"2b_df_multiview_action_conditioned_448_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            wandb_mode="disabled",
        ),
        trainer=dict(
            max_iter=100,
            logging_iter=10,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=10, do_x0_prediction=False, guidance=[0, 3, 7], fps=16, save_s3=False),
                every_n_sample_ema=dict(every_n=10, do_x0_prediction=False, guidance=[0, 3, 7], fps=16, save_s3=False),
            ),
        ),
        checkpoint=dict(
            save_iter=10,
        ),
    ),
    flags={"allow_objects": True},
)


"""
Smoke Test for AgiBotWorld Multi-view 3-camera action-conditioned training
Resolution: 480x1920 (3 views of 480x640 concatenated along width)
Action dimension: 36 (pre-scaled to [0, 1])
Uses smaller dataset from assets/action_conditioned/concat_view/agibot/annotation

torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=ac_reason_embeddings_rectified_flow_2b_agibot_480_1920_smoke ~dataloader_train.dataloaders
"""
ac_reason_embeddings_rectified_flow_2b_agibot_480_1920_smoke = LazyDict(
    dict(
        defaults=[
            "ac_reason_embeddings_rectified_flow_2b_agibot_480_1920",  # Inherit from the full experiment
            {"override /data_train": "agibot_multiview_13frame_480_1920_smoke_train"},
            {"override /data_val": "agibot_multiview_13frame_480_1920_smoke_val"},
            "_self_",
        ],
        job=dict(
            name=f"2b_agibot_multiview_action_conditioned_480_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            wandb_mode="disabled",
        ),
        trainer=dict(
            max_iter=100,
            logging_iter=10,
            callbacks=dict(
                every_n_sample_reg=dict(every_n=10, do_x0_prediction=False, guidance=[0, 3, 7], fps=16, save_s3=False),
                every_n_sample_ema=dict(every_n=10, do_x0_prediction=False, guidance=[0, 3, 7], fps=16, save_s3=False),
            ),
        ),
        checkpoint=dict(
            save_iter=10,
        ),
    ),
    flags={"allow_objects": True},
)


# Map of experiment configs to their static registration names (used by Hydra)
# The actual job.name will have timestamps, but Hydra needs static names
experiments = {
    ac_reason_embeddings_rectified_flow_2b_256_320_df: "ac_reason_embeddings_rectified_flow_2b_256_320_df",
    ac_reason_embeddings_rectified_flow_2b_multiview_448_1344: "ac_reason_embeddings_rectified_flow_2b_multiview_448_1344",
    ac_reason_embeddings_rectified_flow_2b_agibot_480_1920: "ac_reason_embeddings_rectified_flow_2b_agibot_480_1920",
    ac_reason_embeddings_rectified_flow_2b_multiview_448_1344_smoke: "ac_reason_embeddings_rectified_flow_2b_multiview_448_1344_smoke",
    ac_reason_embeddings_rectified_flow_2b_agibot_480_1920_smoke: "ac_reason_embeddings_rectified_flow_2b_agibot_480_1920_smoke",
}


for config, static_name in experiments.items():
    cs.store(group="experiment", package="_global_", name=static_name, node=config)

