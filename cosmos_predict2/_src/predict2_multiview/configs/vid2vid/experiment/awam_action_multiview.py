# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Action-conditioned multiview experiments (AWAM).

from hydra.core.config_store import ConfigStore

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_predict2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)


def awam_action_multiview_2b_3views_448() -> dict:
    state_t = 1 + 12 // 4
    return dict(
        defaults=[
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {"override /callbacks": ["basic", "wandb", "cluster_speed"]},
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            {"override /data_train": "avla_action_multiview_13frame_448_448_train"},
            {"override /data_val": "avla_action_multiview_13frame_448_448_val"},
            {"override /conditioner": "video_prediction_multiview_action_conditioner"},
            {"override /model": "fsdp_rectified_flow_multiview"},
            {"override /net": "cosmos_v1_2B_multiview_action"},
            "_self_",
        ],
        job=dict(
            group="awam",
            name="awam_action_multiview_2b_3views_448",
            wandb_mode="online",
        ),
        optimizer=dict(
            lr=3e-5,
            weight_decay=1e-3,
            betas=[0.9, 0.999],
        ),
        checkpoint=dict(
            load_from_object_store=dict(enabled=False),
            save_to_object_store=dict(enabled=False),
            save_iter=1_000,
        ),
        trainer=dict(
            max_iter=10_000,
            logging_iter=10,
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1_000,
                    do_x0_prediction=False,
                    guidance=[0, 3, 7],
                    fps=16,
                    save_s3=False,
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1_000,
                    do_x0_prediction=False,
                    is_ema=True,
                    guidance=[0, 3, 7],
                    fps=16,
                    save_s3=False,
                ),
                heart_beat=dict(
                    save_s3=False,
                ),
                iter_speed=dict(
                    hit_thres=5,
                    every_n=1,
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
                min_num_conditional_frames=0,
                max_num_conditional_frames=2,
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                condition_locations=["first_random_n"],
                fsdp_shard_size=1,
                resolution="720p",
                state_t=state_t,
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="uniform",
                train_time_distribution="logitnormal",
                online_text_embeddings_as_dict=False,
                net=dict(
                    concat_view_embedding=True,
                    view_condition_dim=6,
                    state_t=state_t,
                    n_cameras_emb=3,
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=1.0,
                    rope_w_extrapolation_ratio=1.0,
                    rope_t_extrapolation_ratio=float(state_t) / 24,
                    timestep_scale=0.001,
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    use_wan_fp32_strategy=True,
                    action_dim=7,
                    num_action_per_chunk=12,
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.0,
                        use_empty_string=False,
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                ),
            ),
        ),
    )


experiments = [
    awam_action_multiview_2b_3views_448(),
]

cs = ConfigStore.instance()

for _item in experiments:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
