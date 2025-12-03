"""
Prepare data for emotion detection model training
- Splits dataset into train/val/test
- Calculates class weights
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
import os

# Create processed_data directory if it doesn't exist
os.makedirs('processed_data', exist_ok=True)

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('dataset/data/legend.csv')
print(f"Total samples: {len(df)}")
print(f"\nClass distribution:")
print(df['emotion'].value_counts())

# Convert all emotion labels to lowercase to merge categories like 'happiness' and 'HAPPINESS'
df['emotion'] = df['emotion'].str.lower()

# Verify the new distribution
print("New Class Distribution:")
print(df['emotion'].value_counts())

# First split: 80% train+val, 20% test
train_val_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df['emotion']
)

# Second split: 80% train, 20% val (from train+val)
train_df, val_df = train_test_split(
    train_val_df, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_val_df['emotion']
)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_df)}")
print(f"  Val: {len(val_df)}")
print(f"  Test: {len(test_df)}")

# Save splits
train_df.to_csv('processed_data/train.csv', index=False)
val_df.to_csv('processed_data/val.csv', index=False)
test_df.to_csv('processed_data/test.csv', index=False)
print("\nSaved train/val/test CSV files to processed_data/")

# Calculate class weights
classes = sorted(df['emotion'].unique())
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(classes),
    y=train_df['emotion']
)

# Create class weight dictionary
class_weight_dict = {cls: float(weight) for cls, weight in zip(classes, class_weights)}

# Save class weights
with open('processed_data/class_weights.json', 'w') as f:
    json.dump(class_weight_dict, f, indent=2)

print("\nClass weights:")
for cls, weight in class_weight_dict.items():
    print(f"  {cls}: {weight:.2f}")

print("\n✓ Data preparation complete!")
