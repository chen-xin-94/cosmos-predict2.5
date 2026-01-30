#!/bin/bash
# Training script for multi-view action-conditioned video generation
# 3 cameras (left, top, wrist) concatenated along width: 448x1344

set -e

# Set output directory for checkpoints
export IMAGINAIRE_OUTPUT_ROOT=/raid/chen.xin/repo/cosmos-predict2.5/imaginaire-output

# Set visible GPU(s)
export CUDA_VISIBLE_DEVICES=1,4,5,7

# Run training
torchrun --nproc_per_node=4 --master_port=$((12000 + RANDOM % 10000)) \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py \
    -- experiment=ac_reason_embeddings_rectified_flow_2b_multiview_448_1344 \
    ~dataloader_train.dataloaders \
    checkpoint.load_training_state=True
    
echo "Training completed!"