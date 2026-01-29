# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration for AgiBotWorld dataset preprocessing.
Adapted from: /raid/chen.xin/repo/any4lerobot/agibot2lerobot/agibot_utils/config.py
"""

# State keys to extract and flatten (in order)
# Dimensions: 2 + 8 + 6 + 2 + 14 + 14 + 4 + 3 + 2 = 55
STATE_KEYS = [
    "effector.position",    # shape (2,)
    "end.orientation",      # shape (2, 4) -> 8
    "end.position",         # shape (2, 3) -> 6
    "head.position",        # shape (2,)
    "joint.current_value",  # shape (14,)
    "joint.position",       # shape (14,)
    "robot.orientation",    # shape (4,)
    "robot.position",       # shape (3,)
    "waist.position",       # shape (2,)
]

# Action keys to extract and flatten (in order)
# Dimensions: 2 + 8 + 6 + 2 + 14 + 2 + 2 = 36
ACTION_KEYS = [
    "effector.position",    # shape (2,)
    "end.orientation",      # shape (2, 4) -> 8
    "end.position",         # shape (2, 3) -> 6
    "head.position",        # shape (2,)
    "joint.position",       # shape (14,)
    "robot.velocity",       # shape (2,)
    "waist.position",       # shape (2,)
]

# Video views to extract (in order)
VIDEO_VIEWS = [
    "hand_left_color",
    "head_color",
    "hand_right_color",
]

# Task type configurations (gripper is default for most tasks)
DEXHAND_TASK_IDS = [
    475, 536, 547, 548, 549, 554, 577, 578, 591, 595,
    608, 620, 622, 660, 679, 705, 710, 727, 730, 731, 749, 753,
]

TACTILE_TASK_IDS = [
    666, 675, 676, 677, 694, 737, 774,
]


def get_task_type(task_id: int) -> str:
    """Determine task type based on task ID."""
    if task_id in DEXHAND_TASK_IDS:
        return "dexhand"
    elif task_id in TACTILE_TASK_IDS:
        return "tactile"
    else:
        return "gripper"
