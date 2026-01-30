#!/bin/bash

# ============================================================================
# Original Single-View Action-Conditioned Inference
# ============================================================================

CHECKPOINTS_DIR=/raid/yusong.li/workspace/cosmos-predict2.5_df/imaginaire-output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_df_action_w_text_conditioned_10k_bs4/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

export CUDA_VISIBLE_DEVICES=3

python examples/action_conditioned.py \
  -i cosmos_predict2/datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand/inference_params.json \
  -o outputs/action_conditioned/df_franka_single_arm_bf16_w_new_text_12.5k_novel_image2 \
  --experiment ac_reason_embeddings_rectified_flow_2b_256_320_avla \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt


# ============================================================================
# Multiview Action-Conditioned Inference (awam_action_multiview_2b_3views_448)
# ============================================================================

export CUDA_VISIBLE_DEVICES=1

python examples/action_conditioned.py \
  -i assets/action_conditioned/concat_view/inference_params.json \
  -o outputs/awam_multiview_3views_448 \
  --experiment awam_action_multiview_2b_3views_448 \
  --checkpoint-path imaginaire-output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_df_multiview_action_conditioned_448_20260123_190323/checkpoints/iter_000010000/model_ema_bf16.pt \
  --config-file cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py


# ============================================================================
# Multiview Action-Conditioned Inference
# Experiment: ac_reason_embeddings_rectified_flow_2b_multiview_448_1344
# ============================================================================

CHECKPOINTS_DIR=imaginaire-output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_df_multiview_action_conditioned_448_20260123_190323/checkpoints
CHECKPOINT_ITER=iter_000050000  # Or use: $(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

export CUDA_VISIBLE_DEVICES=1

python examples/action_conditioned.py \
  -i assets/action_conditioned/concat_view/inference_params.json \
  -o outputs/ac_mv_448_1344_${CHECKPOINT_ITER} \
  --experiment ac_reason_embeddings_rectified_flow_2b_multiview_448_1344 \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt
