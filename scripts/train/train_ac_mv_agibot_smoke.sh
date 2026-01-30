#!/bin/bash
# Training script for smoke test of AgiBotWorld multi-view action-conditioned video generation
# 3 cameras (hand_left, head, hand_right) concatenated along width: 480x1920
# Uses smaller dataset for quick testing

set -e

# Set output directory for checkpoints
export IMAGINAIRE_OUTPUT_ROOT=/raid/chen.xin/repo/cosmos-predict2.5/imaginaire-output

# Set visible GPU(s)
export CUDA_VISIBLE_DEVICES=2

# Run training
torchrun --nproc_per_node=1 --master_port=$((12000 + RANDOM % 10000)) \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_agibot_480_1920_smoke \
    ~dataloader_train.dataloaders

echo "Smoke test training completed!"
