export IMAGINAIRE_OUTPUT_ROOT=/raid/yusong.li/workspace/cosmos-predict2.5_df/imaginaire-output

export CUDA_VISIBLE_DEVICES=2

torchrun --nproc_per_node=1 --master_port=$((12000 + RANDOM % 10000)) -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_avla ~dataloader_train.dataloaders