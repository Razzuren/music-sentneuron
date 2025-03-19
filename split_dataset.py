import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Dataset Path
# midi_encoder.py --path datasets/split/train --transp 10 --strech 10
# midi_encoder.py --path datasets/split/test.txt --transp 10 --strech 10
DATASET_PATH = "./midis"
SAVE_PATH = "./datasets/split"
os.makedirs(SAVE_PATH, exist_ok=True)

# Load dataset and group by quadrant
midi_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".mid") or f.endswith(".midi")]
file_quadrants = []

for file in midi_files:
    quadrant = int(file.split("_")[0][1]) - 1  # Assuming filenames start with quadrant info (e.g., Q1_xxx.mid)
    file_quadrants.append((file, quadrant))

# Convert to DataFrame
data = pd.DataFrame(file_quadrants, columns=["file_name", "quadrant"])

# Ensure balanced quadrant distribution
train_files, test_files = train_test_split(data, test_size=0.2, stratify=data["quadrant"], random_state=42)

# Move files to respective directories
train_path = os.path.join(SAVE_PATH, "train")
test_path = os.path.join(SAVE_PATH, "test")
os.makedirs(train_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

for _, row in train_files.iterrows():
    os.rename(os.path.join(DATASET_PATH, row["file_name"]), os.path.join(train_path, row["file_name"]))

for _, row in test_files.iterrows():
    os.rename(os.path.join(DATASET_PATH, row["file_name"]), os.path.join(test_path, row["file_name"]))

print("✅ MIDI files successfully split into train and test datasets!")