"""
Configuration file for emotion detection project
"""

import torch

# Paths
IMAGES_PATH = 'dataset/images'
TRAIN_CSV = 'processed_data/train.csv'
VAL_CSV = 'processed_data/val.csv'
TEST_CSV = 'processed_data/test.csv'
CLASS_WEIGHTS_PATH = 'processed_data/class_weights.json'
MODELS_DIR = 'src/models'
RESULTS_DIR = 'results'

# Training hyperparameters
BATCH_SIZE = 64
IMG_SIZE = 224
NUM_WORKERS = 2
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15

# Model hyperparameters
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.5

# Data augmentation
USE_IMAGENET_STATS = True
USE_BALANCED_SAMPLING = True

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random seed for reproducibility
RANDOM_SEED = 42

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Custom normalization (based on your data)
CUSTOM_MEAN = [119.18/255.0] * 3
CUSTOM_STD = [22.43/255.0] * 3