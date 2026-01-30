

# Set output directory for checkpoints
export IMAGINAIRE_OUTPUT_ROOT=/raid/chen.xin/repo/cosmos-predict2.5/imaginaire-output

# Set visible GPU(s)
export CUDA_VISIBLE_DEVICES=5,7

torchrun --nproc_per_node=2 --master_port=$((12000 + RANDOM % 10000)) -m scripts.train --config=cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py -- experiment=awam_action_multiview_2b_3views_448

echo "Training completed!"