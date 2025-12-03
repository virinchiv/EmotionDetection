import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

class EmotionDataset(Dataset):
    def __init__(self, csv_file, images_path, transform=None, class_to_idx=None):
        """
        Args:
            csv_file: Path to CSV with image names and labels
            images_path: Directory with all images
            transform: Optional transform to be applied on images
            class_to_idx: Dictionary mapping class names to indices
        """
        self.df = pd.read_csv(csv_file)
        self.images_path = images_path
        self.transform = transform
        
        # Create class to index mapping
        if class_to_idx is None:
            self.classes = sorted(self.df['emotion'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(self.images_path, img_name)
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        # Get label
        emotion = self.df.iloc[idx]['emotion']
        label = self.class_to_idx[emotion]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ImageNet normalization stats (for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Custom normalization based on your data analysis
CUSTOM_MEAN = [119.18/255.0] * 3  # Your mean pixel intensity normalized
CUSTOM_STD = [22.43/255.0] * 3    # Your std normalized

def get_transforms(mode='train', img_size=224, use_imagenet_stats=True):
    """
    Get appropriate transforms for train/val/test
    
    Args:
        mode: 'train', 'val', or 'test'
        img_size: Target image size
        use_imagenet_stats: Use ImageNet normalization (for transfer learning)
    """
    
    if use_imagenet_stats:
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    else:
        normalize = transforms.Normalize(mean=CUSTOM_MEAN, std=CUSTOM_STD)
    
    if mode == 'train':
        # Aggressive augmentation for training (especially for minority classes)
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),  # Slightly larger for cropping
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomGrayscale(p=0.1),  # Since 92.9% are already grayscale
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),  # Random erasing augmentation
        ])
    
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


class BalancedBatchSampler(torch.utils.data.Sampler):
    """
    Sampler that oversamples minority classes to balance the dataset
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Get class distribution
        labels = [self.dataset.df.iloc[i]['emotion'] for i in range(len(dataset))]
        self.class_counts = pd.Series(labels).value_counts().to_dict()
        
        # Calculate sampling weights (inverse frequency)
        weights = []
        for i in range(len(dataset)):
            label = self.dataset.df.iloc[i]['emotion']
            weights.append(1.0 / self.class_counts[label])
        
        self.weights = torch.DoubleTensor(weights)
        self.num_samples = len(self.dataset)
    
    def __iter__(self):
        # Sample with replacement based on weights
        indices = torch.multinomial(self.weights, self.num_samples, replacement=True)
        return iter(indices.tolist())
    
    def __len__(self):
        return self.num_samples


def create_dataloaders(images_path, train_csv, val_csv, test_csv, 
                       batch_size=32, img_size=224, use_imagenet_stats=True,
                       use_balanced_sampling=True, num_workers=4):
    """
    Create train, validation, and test dataloaders
    """
    
    # Get transforms
    train_transform = get_transforms('train', img_size, use_imagenet_stats)
    val_transform = get_transforms('val', img_size, use_imagenet_stats)
    test_transform = get_transforms('test', img_size, use_imagenet_stats)
    
    # Create datasets
    train_dataset = EmotionDataset(train_csv, images_path, transform=train_transform)
    val_dataset = EmotionDataset(val_csv, images_path, transform=val_transform, 
                                 class_to_idx=train_dataset.class_to_idx)
    test_dataset = EmotionDataset(test_csv, images_path, transform=test_transform,
                                  class_to_idx=train_dataset.class_to_idx)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    print(f"  Classes: {train_dataset.classes}")
    
    # Create dataloaders
    if use_balanced_sampling:
        # Use balanced batch sampler for training
        train_sampler = BalancedBatchSampler(train_dataset, batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, 
                                 num_workers=num_workers, pin_memory=True)
        print("Using balanced batch sampling for training")
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=num_workers, pin_memory=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader, train_dataset.class_to_idx

# Initialize dataloaders
images_path = 'dataset/images'
train_csv = 'processed_data/train.csv'
val_csv = 'processed_data/val.csv'
test_csv = 'processed_data/test.csv'

train_loader, val_loader, test_loader, class_to_idx = create_dataloaders(
    images_path, train_csv, val_csv, test_csv,
    batch_size=64,
    img_size=224,
    use_imagenet_stats=True,  # True for transfer learning models
    use_balanced_sampling=True,
    num_workers=2
)


def visualize_augmentations(dataset, num_samples=8):
    """Show original and augmented versions of samples"""
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 5))
    
    # Get a sample image
    idx = np.random.randint(len(dataset))
    
    for i in range(num_samples):
        img, label = dataset[idx]
        
        # Denormalize for visualization
        img_np = img.numpy().transpose(1, 2, 0)
        img_np = img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        img_np = np.clip(img_np, 0, 1)
        
        row = i // num_samples
        col = i % num_samples
        axes[row, col].imshow(img_np)
        axes[row, col].axis('off')
        if i == 0:
            axes[row, col].set_title(f"Class: {dataset.idx_to_class[label]}")
    
    plt.suptitle("Data Augmentation Examples (same image, different augmentations)")
    plt.tight_layout()
    plt.show()

# Visualize
train_dataset_vis = EmotionDataset(train_csv, images_path, 
                                   transform=get_transforms('train', 224, True))
visualize_augmentations(train_dataset_vis, num_samples=8)



class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Focuses more on hard-to-classify examples
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights (tensor of size num_classes)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Load class weights
with open('processed_data/class_weights.json', 'r') as f:
    class_weight_dict = json.load(f)

# Convert to tensor
classes_sorted = sorted(class_to_idx.keys())
class_weights = torch.tensor([class_weight_dict[cls] for cls in classes_sorted], 
                            dtype=torch.float32).to(device)

print("Class weights loaded:")
for cls, weight in zip(classes_sorted, class_weights):
    print(f"  {cls}: {weight:.2f}")