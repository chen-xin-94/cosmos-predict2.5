"""
Split processed JSONs into train/val/test.
"""

import argparse
import os
import random
import shutil

parser = argparse.ArgumentParser(description="Split processed JSONs into train/val/test.")
parser.add_argument(
    "--processed-dir",
    type=str,
    default="datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand/annotation/all",
    help="Path to the directory containing processed JSON files.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=28,
    help="Random seed for shuffling files.",
)
args = parser.parse_args()

processed_dir = args.processed_dir
parent_dir = os.path.dirname(processed_dir)

# Create split directories
train_dir = os.path.join(parent_dir, "train")
val_dir   = os.path.join(parent_dir, "val")
test_dir  = os.path.join(parent_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Collect all JSON files (exclude directories)
all_files = [f for f in os.listdir(processed_dir) if f.endswith(".json")]

total = len(all_files)
print(f"Found {total} JSON files in {processed_dir}.")

# Set random seed for reproducibility
random.seed(args.seed)

# Shuffle files
random.shuffle(all_files)

# Split sizes
train_count = int(0.90 * total)
val_count   = int(0.05 * total)
test_count  = total - train_count - val_count  # remaining

print(f"Train: {train_count}, Val: {val_count}, Test: {test_count}")

# Assignment
train_files = all_files[:train_count]
val_files   = all_files[train_count:train_count + val_count]
test_files  = all_files[train_count + val_count:]

# Move operation
def move_files(files, target_dir):
    for fname in files:
        src = os.path.join(processed_dir, fname)
        dst = os.path.join(target_dir, fname)
        # shutil.copy(src, dst)  # use copy (safer)
        shutil.move(src, dst) 

move_files(train_files, train_dir)
move_files(val_files, val_dir)
move_files(test_files, test_dir)

print("Done! train/, val/, test/ created in {parent_dir}")
