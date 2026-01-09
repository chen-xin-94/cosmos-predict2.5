import os
import random
import shutil

# processed_json directory
processed_dir = "/raid/yusong/workspace/cosmos-predict2.5_df/cosmos_predict2/datasets/df/avla_nov_8_merged_per_embodiment_2025-11-12/fr3_single_arm_franka_hand/annotation/processed_json"

# Parent directory (annotation/)
parent_dir = os.path.dirname(processed_dir)

# Create split directories parallel to processed_json/
train_dir = os.path.join(parent_dir, "train")
val_dir   = os.path.join(parent_dir, "val")
test_dir  = os.path.join(parent_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Collect all JSON files (exclude directories)
all_files = [f for f in os.listdir(processed_dir) if f.endswith(".json")]

total = len(all_files)
print(f"Found {total} JSON files in processed_json/.")

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
        shutil.copy(src, dst)  # use copy (safer)
        # shutil.move(src, dst)  # use move if you want to REMOVE from processed_json

move_files(train_files, train_dir)
move_files(val_files, val_dir)
move_files(test_files, test_dir)

print("Done! train/, val/, test/ created PARALLEL to processed_json/")
