#!/usr/bin/env python3
"""Test script to verify multi-view dataset loading works correctly.

Usage:
    cd /raid/chen.xin/repo/cosmos-predict2.5
    source .venv/bin/activate
    PYTHONPATH=. python scripts/test_multiview_dataset.py
"""

import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cosmos_predict2._src.predict2.action.datasets.dataset_df import ActionConditionedMultiViewDataset_DF, Dataset_3D_DF


def test_datasets():
    # Base paths
    base_path = "datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand"
    annotation_path = os.path.join(base_path, "annotation")
    
    train_path = os.path.join(annotation_path, "train")
    val_path = os.path.join(annotation_path, "val")
    test_path = os.path.join(annotation_path, "test")
    
    print("=" * 60)
    print("Testing Multi-View Dataset Loading")
    print("=" * 60)
    
    # Common parameters
    common_params = dict(
        train_annotation_path=train_path,
        val_annotation_path=val_path,
        test_annotation_path=test_path,
        video_path="",  # Not used for df datasets (absolute paths in JSON)
        fps_downsample_ratio=1,
        num_action_per_chunk=12,  # 13 frames total
        accumulate_action=False,
        video_size=[256, 320],  # H, W
        val_start_frame_interval=100,
        debug=True,  # Only load first 10 samples
        normalize=False,
        mode="val",
        state_key="state",
        gripper_key="continuous_gripper_state",
        text_key="text",
    )
    
    # ==========================================
    # Test 1: Single-view Dataset_3D_DF (backward compatibility)
    # ==========================================
    print("\n" + "-" * 60)
    print("Test 1: Single-view Dataset_3D_DF (cam_ids=[0])")
    print("-" * 60)
    
    start_time = time.time()
    single_view_dataset = Dataset_3D_DF(
        **common_params,
        cam_ids=[0],  # Only first camera (frame_camera_left)
    )
    print(f"Dataset created in {time.time() - start_time:.2f}s")
    print(f"Number of samples: {len(single_view_dataset)}")
    
    # Load a sample
    start_time = time.time()
    sample_sv = single_view_dataset[0]
    print(f"Sample loaded in {time.time() - start_time:.2f}s")
    
    video_sv = sample_sv["video"]
    action_sv = sample_sv["action"]
    
    print(f"Video shape: {video_sv.shape}")  # Expected: [C, T, H, W]
    print(f"Video dtype: {video_sv.dtype}")
    print(f"Action shape: {action_sv.shape}")  # Expected: [T-1, 7]
    print(f"Action dtype: {action_sv.dtype}")
    
    single_view_width = video_sv.shape[3]
    print(f"Single view width: {single_view_width}")
    
    # ==========================================
    # Test 2: Multi-view ActionConditionedMultiViewDataset_DF (3 views)
    # ==========================================
    print("\n" + "-" * 60)
    print("Test 2: Multi-view ActionConditionedMultiViewDataset_DF (cam_ids=[0, 1, 2])")
    print("-" * 60)
    
    start_time = time.time()
    multi_view_dataset = ActionConditionedMultiViewDataset_DF(
        **common_params,
        cam_ids=[0, 1, 2],  # All 3 cameras
    )
    print(f"Dataset created in {time.time() - start_time:.2f}s")
    print(f"Number of samples: {len(multi_view_dataset)}")
    
    # Load a sample
    start_time = time.time()
    sample_mv = multi_view_dataset[0]
    print(f"Sample loaded in {time.time() - start_time:.2f}s")
    
    video_mv = sample_mv["video"]
    action_mv = sample_mv["action"]
    
    print(f"Video shape: {video_mv.shape}")  # Expected: [C, T, H, W*3]
    print(f"Video dtype: {video_mv.dtype}")
    print(f"Action shape: {action_mv.shape}")  # Expected: [T-1, 7]
    print(f"Action dtype: {action_mv.dtype}")
    
    multi_view_width = video_mv.shape[3]
    print(f"Multi-view width: {multi_view_width}")
    
    # ==========================================
    # Test 3: Verify width concatenation
    # ==========================================
    print("\n" + "-" * 60)
    print("Test 3: Width Concatenation Verification")
    print("-" * 60)
    
    expected_width = single_view_width * 3
    print(f"Expected multi-view width: {single_view_width} * 3 = {expected_width}")
    print(f"Actual multi-view width: {multi_view_width}")
    
    if multi_view_width == expected_width:
        print("✅ Width concatenation CORRECT!")
    else:
        print("❌ Width concatenation FAILED!")
        print(f"   Expected: {expected_width}, Got: {multi_view_width}")
    
    # ==========================================
    # Test 4: Test 2-view configuration
    # ==========================================
    print("\n" + "-" * 60)
    print("Test 4: Two-view configuration (cam_ids=[0, 2])")
    print("-" * 60)
    
    two_view_dataset = ActionConditionedMultiViewDataset_DF(
        **common_params,
        cam_ids=[0, 2],  # left + wrist
    )
    
    sample_2v = two_view_dataset[0]
    video_2v = sample_2v["video"]
    
    print(f"Two-view video shape: {video_2v.shape}")
    expected_2v_width = single_view_width * 2
    actual_2v_width = video_2v.shape[3]
    
    if actual_2v_width == expected_2v_width:
        print("✅ Two-view width CORRECT!")
    else:
        print("❌ Two-view width FAILED!")
    
    # ==========================================
    # Test 5: Load multiple samples
    # ==========================================
    print("\n" + "-" * 60)
    print("Test 5: Loading multiple samples from multi-view dataset")
    print("-" * 60)
    
    for i in range(min(3, len(multi_view_dataset))):
        start_time = time.time()
        sample = multi_view_dataset[i]
        load_time = time.time() - start_time
        print(f"Sample {i}: video={sample['video'].shape}, action={sample['action'].shape}, time={load_time:.2f}s")
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Single-view video shape: {video_sv.shape}")
    print(f"Multi-view (3) video shape: {video_mv.shape}")
    print(f"Two-view video shape: {video_2v.shape}")
    print(f"Action shape (same for all): {action_sv.shape}")
    print("\nAll tests completed!")


if __name__ == "__main__":
    test_datasets()
