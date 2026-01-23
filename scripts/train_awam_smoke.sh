

# Set output directory for checkpoints
export IMAGINAIRE_OUTPUT_ROOT=/raid/chen.xin/repo/cosmos-predict2.5/imaginaire-output

# remove previous run
# rm -r $IMAGINAIRE_OUTPUT_ROOT/cosmos_predict2_multiview/awam/awam_action_multiview_2b_3views_448

# Set visible GPU(s)
export CUDA_VISIBLE_DEVICES=0

# DATASET SELECTION:
# change `base_path` in function `register_awam_dataloader` `cosmos_predict2/_src/predict2_multiview/configs/vid2vid/defaults/dataloader_local.py`


torchrun --nproc_per_node=1 --master_port=$((12000 + RANDOM % 10000)) -m scripts.train --config=cosmos_predict2/_src/predict2_multiview/configs/vid2vid/config.py -- experiment=awam_action_multiview_2b_3views_448 \
  job.wandb_mode=disabled \
  checkpoint.save_iter=10 \
  trainer.callbacks.every_n_sample_reg.every_n=10 \
  trainer.callbacks.every_n_sample_ema.every_n=10

