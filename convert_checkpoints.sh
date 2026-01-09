
CHECKPOINTS_DIR=/raid/yusong.li/workspace/cosmos-predict2.5_df/imaginaire-output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_avla_action_w_text_conditioned_10k_bs4/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
# python ./scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR

export CUDA_VISIBLE_DEVICES=3

python examples/action_conditioned.py \
-i cosmos_predict2/datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand/inference_params.json \
-o outputs/action_conditioned/avla_franka_single_arm_bf16_w_new_text_12.5k_novel_image2 \
--experiment ac_reason_embeddings_rectified_flow_2b_256_320_avla \
--checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
