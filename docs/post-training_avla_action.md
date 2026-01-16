# Note 02.12.2025

Data preprocessing scripts are under `avla_data_preprocessing/`

Training annotations (json files) are stored under `cosmos_predict2/datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand/annotation/`
The full episodes are recorded. During training, set `fps_downsample_ratio=6` to simulate training with `FPS=5` for consistency with bridge dataset.

Class `Dataset_3D_df` is defined in `cosmos_predict2/_src/predict2/action/datasets/dataset_local.py`. The experiment configurations are defined in `cosmos_predict2/experiments/base/action.py`.

It appears that action-conditioned training requires no prompt (As in original post-training pipeline). Modified dataloader to also enable training with text.



## 1. Data Preprocessing
Create json files with specific format with `avla_data_preprocessing/get_json_from_par_batch.py`.
Create train-val-test split with `avla_data_preprocessing/split.py`.

## 2. Training Arguments
Currently, all training arguments and configs are defined in Dict `ac_reason_embeddings_rectified_flow_2b_256_320_df` under `cosmos_predict2/experiments/base/action.py`.


For now, `bash_path` in [`cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py`](../cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py#L33) need to be manually modified if using other data from data_foundry.

(now set default as `cosmos_predict2/datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand`)


## 3. Training
Run

```
bash ac_train.sh
```

where the exact commands are

```
export IMAGINAIRE_OUTPUT_ROOT=/raid/.../cosmos-predict2.5/imaginaire-output
export CUDA_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 --master_port=$((12000 + RANDOM % 10000)) -m scripts.train --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py  -- experiment=ac_reason_embeddings_rectified_flow_2b_256_320_avla ~dataloader_train.dataloaders
```
With this setting, the checkpoint will be saved under `cosmos-predict2.5/imaginaire-output`


## 4. Convert checkpoint and inference
After the checkpoint is saved, it need to be converted to PyTorch format(`convert_checkpoints.sh`):
```
CHECKPOINTS_DIR=/raid/yusong.li/workspace/cosmos-predict2.5_df/imaginaire-output/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/2b_avla_action_w_text_conditioned_10k_bs4/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python ./scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```
Refer to `post-training_video2world_action.md` for a detailed description of checkpoint format.


Inference:
```
export CUDA_VISIBLE_DEVICES=0

python examples/action_conditioned.py \
-i cosmos_predict2/datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand/inference_params.json \
-o outputs/action_conditioned/avla_franka_single_arm_bf16_w_new_text_12.5k_novel_image2 \
--experiment ac_reason_embeddings_rectified_flow_2b_256_320_avla \
--checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
```

`inference_params.json` includes all inference setting and hyperparameters:
```
{
  "name": "avla_franka_single_arm_bf16_w_text_12.5k_debug",
  "input_root": "/raid/yusong.li/workspace/cosmos-predict2.5_df/cosmos_predict2/datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand",
  "input_json_sub_folder": "annotation/test",
  "save_root": "outputs/action_conditioned/avla_franka_single_arm_bf16_w_text_12.5k_debug",
  "guidance": 0,
  "resolution": "256,320",
  "camera_id": 0,
  "start": 0,
  "end": 10,
  "fps_downsample_ratio": 6,
  "gripper_scale": 1.0,
  "gripper_key": "continuous_gripper_state",
  "state_key": "state",
  "reverse": false,
  "single_chunk": false,
  "start_frame_idx": 0,
  "save_fps": 20,
  "num_latent_conditional_frames": 1,
  "action_scaler": 20.0,
  "use_quat": false,
  "action_load_fn": "cosmos_predict2.action_conditioned.load_default_action_fn",
  "negative_prompt": "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.",
  "seed": 42,
  "prompt": null,
  "use_text": true
}

```







