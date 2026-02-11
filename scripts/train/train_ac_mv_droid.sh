#!/bin/bash
# Training script for DROID multi-view action-conditioned video generation
# 3 cameras concatenated along width: 176x960

set -e

# Set output directory for checkpoints
export IMAGINAIRE_OUTPUT_ROOT=/raid/chen.xin/repo/cosmos-predict2.5/imaginaire-output

# Set visible GPU(s)
export CUDA_VISIBLE_DEVICES=1,2

# Run training
torchrun --nproc_per_node=2 --master_port=$((12000 + RANDOM % 10000)) \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_droid_176_960 \
    ~dataloader_train.dataloaders

echo "Training completed!"
