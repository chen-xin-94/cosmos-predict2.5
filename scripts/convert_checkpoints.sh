#!/bin/bash

# ============================================================================
# Original Single-View Action-Conditioned Checkpoint Conversion
# ============================================================================

CHECKPOINTS_DIR=/raid/yusong.li/workspace/cosmos-predict2.5_df/imaginaire-output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_df_action_w_text_conditioned_10k_bs4/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
# python ./scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR


# ============================================================================
# Multiview Action-Conditioned Checkpoint Conversion
# Experiment: awam_action_multiview_2b_3views_448
# ============================================================================

AWAM_CHECKPOINT_DIR=imaginaire-output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_df_multiview_action_conditioned_448_20260123_190323/checkpoints/iter_000010000

# Convert DCP checkpoint to PyTorch format
# python ./scripts/convert_distcp_to_pt.py $AWAM_CHECKPOINT_DIR/model $AWAM_CHECKPOINT_DIR


# ============================================================================
# Multiview Action-Conditioned Checkpoint Conversion
# Experiment: ac_reason_embeddings_rectified_flow_2b_multiview_448_1344
# ============================================================================

CHECKPOINTS_DIR=imaginaire-output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_df_multiview_action_conditioned_448_20260123_190323/checkpoints
CHECKPOINT_ITER=iter_000050000  # Or use: $(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python ./scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
