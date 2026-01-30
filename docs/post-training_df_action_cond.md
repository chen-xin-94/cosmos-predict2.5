# Note 02.12.2025

Data preprocessing scripts are under `df_data_preprocessing/`

Training annotations (json files) are stored under `datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand/annotation/`
The full episodes are recorded. During training, set `fps_downsample_ratio=6` to simulate training with `FPS=5` for consistency with bridge dataset.

Class `Dataset_3D_DF` is defined in `cosmos_predict2/_src/predict2/action/datasets/dataset_local.py`. The experiment configurations are defined in `cosmos_predict2/experiments/base/action.py`.

It appears that action-conditioned training requires no prompt (As in original post-training pipeline). Modified dataloader to also enable training with text.



## 1. Data Preprocessing
Create json files with specific format with `scripts/preprocessing/preprocess_df.py`.
Create train-val-test split with `scripts/preprocessing/split.py`.

## 2. Training Arguments
Currently, all training arguments and configs are defined in Dict `ac_reason_embeddings_rectified_flow_2b_256_320_df` under `cosmos_predict2/experiments/base/action.py`.


For now, `bash_path` in [`cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py`](../cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py#L33) need to be manually modified if using other data from data_foundry.

(now set default as `datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand`)


## 3. Training
Run

```
bash ac_train.sh
```

where the exact commands are

```
export IMAGINAIRE_OUTPUT_ROOT=imaginaire-output
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 --master_port=$((12000 + RANDOM % 10000)) -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_df ~dataloader_train.dataloaders
```
With this setting, the checkpoint will be saved under `imaginaire-output`


## 4. Convert checkpoint and inference

### 4.1 Converting DCP checkpoint to consolidated PyTorch format
After the checkpoint is saved, it need to be converted to PyTorch format (`scripts/convert_checkpoints.sh`):
```
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_df_action_w_text_conditioned_10k_bs4/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python ./scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```
Refer to `post-training_video2world_action.md` for a detailed description of checkpoint format.


### 4.2 Running inference (single-view)
Use `scripts/inference.sh` (single-view block) or run:
```
export CUDA_VISIBLE_DEVICES=0

python examples/action_conditioned.py \
-i assets/action_conditioned/basic/df/inference_params.json \
-o outputs/action_conditioned/df_franka_single_arm_bf16_w_new_text_12.5k_novel_image2 \
--experiment ac_reason_embeddings_rectified_flow_2b_256_320_df \
--checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
```

`inference_params.json` includes all inference setting and hyperparameters:

```







