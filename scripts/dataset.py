"""
PyTorch Dataset and DataLoader for Semiconductor Defect Images
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATASET_CONFIG, AUGMENTATION_CONFIG, TRAINING_CONFIG


class DefectDataset(Dataset):
    """Dataset class for semiconductor defect images"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        normalize: bool = True
    ):
        """
        Args:
            data_dir: Root directory containing split folders
            split: One of 'train', 'val', 'test'
            transform: Albumentations transform pipeline
            normalize: Whether to apply normalization
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.class_names = DATASET_CONFIG['class_names']
        self.num_classes = DATASET_CONFIG['num_classes']
        
        # Create class to index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load all image paths and labels
        self.samples = []
        self._load_samples()
        
        # Calculate dataset statistics for normalization
        if self.normalize and split == 'train':
            self.mean, self.std = self._calculate_statistics()
        else:
            # Use standard values (will be overridden if loading from trained model)
            self.mean = 0.456
            self.std = 0.224
    
    def _load_samples(self):
        """Load all image paths and corresponding labels"""
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            for img_path in class_dir.glob('*.png'):
                self.samples.append({
                    'path': str(img_path),
                    'label': self.class_to_idx[class_name],
                    'class_name': class_name
                })
            
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append({
                    'path': str(img_path),
                    'label': self.class_to_idx[class_name],
                    'class_name': class_name
                })
        
        print(f"Loaded {len(self.samples)} samples for {self.split} split")
    
    def _calculate_statistics(self, sample_size: int = 100) -> Tuple[float, float]:
        """Calculate mean and std of the dataset"""
        print("Calculating dataset statistics...")
        
        # Sample images for efficiency
        sample_indices = np.random.choice(
            len(self.samples),
            min(sample_size, len(self.samples)),
            replace=False
        )
        
        pixel_values = []
        for idx in sample_indices:
            img = cv2.imread(self.samples[idx]['path'], cv2.IMREAD_GRAYSCALE)
            pixel_values.extend(img.flatten() / 255.0)
        
        pixel_values = np.array(pixel_values)
        mean = pixel_values.mean()
        std = pixel_values.std()
        
        print(f"Dataset statistics - Mean: {mean:.3f}, Std: {std:.3f}")
        return mean, std
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample"""
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['path'], cv2.IMREAD_GRAYSCALE)
        
        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Normalize
        if self.normalize:
            image = image.astype(np.float32) / 255.0
            image = (image - self.mean) / self.std
        else:
            image = image.astype(np.float32) / 255.0
        
        # Convert to tensor (add channel dimension)
        image = torch.from_numpy(image).unsqueeze(0).float()
        
        label = sample['label']
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        class_counts = np.zeros(self.num_classes)
        for sample in self.samples:
            class_counts[sample['label']] += 1
        
        # Inverse frequency weighting
        total = class_counts.sum()
        class_weights = total / (self.num_classes * class_counts + 1e-6)
        
        return torch.FloatTensor(class_weights)


def get_transforms(split: str = 'train') -> A.Compose:
    """
    Get albumentations transform pipeline
    
    Args:
        split: One of 'train', 'val', 'test'
    """
    if split == 'train' and AUGMENTATION_CONFIG['train']['enabled']:
        cfg = AUGMENTATION_CONFIG['train']
        
        transforms = [
            # Noise augmentation (simulates SEM imaging artifacts)
            A.GaussNoise(
                var_limit=cfg['gaussian_noise']['var_limit'],
                p=cfg['gaussian_noise']['p']
            ),
            
            # Elastic transform (simulates manufacturing process variations)
            A.ElasticTransform(
                alpha=cfg['elastic_transform']['alpha'],
                sigma=cfg['elastic_transform']['sigma'],
                p=cfg['elastic_transform']['p']
            ),
            
            # Motion blur (simulates optical artifacts)
            A.MotionBlur(
                blur_limit=cfg['motion_blur']['blur_limit'],
                p=cfg['motion_blur']['p']
            ),
            
            # Geometric transformations
            A.RandomRotate90(p=cfg['rotate90']['p']),
            A.HorizontalFlip(p=cfg['horizontal_flip']['p']),
            A.VerticalFlip(p=cfg['vertical_flip']['p']),
            
            # Brightness/Contrast
            A.RandomBrightnessContrast(
                brightness_limit=cfg['brightness_contrast']['brightness_limit'],
                contrast_limit=cfg['brightness_contrast']['contrast_limit'],
                p=cfg['brightness_contrast']['p']
            ),
        ]
    else:
        # No augmentation for val/test
        transforms = []
    
    return A.Compose(transforms)


def create_dataloaders(
    data_dir: str,
    batch_size: int = TRAINING_CONFIG['batch_size'],
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Root directory containing split folders
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = DefectDataset(
        data_dir=data_dir,
        split='train',
        transform=get_transforms('train'),
        normalize=True
    )
    
    val_dataset = DefectDataset(
        data_dir=data_dir,
        split='val',
        transform=get_transforms('val'),
        normalize=True
    )
    val_dataset.mean = train_dataset.mean
    val_dataset.std = train_dataset.std
    
    test_dataset = DefectDataset(
        data_dir=data_dir,
        split='test',
        transform=get_transforms('test'),
        normalize=True
    )
    test_dataset.mean = train_dataset.mean
    test_dataset.std = train_dataset.std
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """Test the dataset and dataloader"""
    from configs.config import SPLITS_DIR
    
    print("Testing Dataset and DataLoader...")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(SPLITS_DIR)
    
    # Test loading a batch
    print("\nTesting batch loading...")
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Label dtype: {labels.dtype}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Print label distribution in batch
    unique, counts = torch.unique(labels, return_counts=True)
    print("\nLabel distribution in batch:")
    for label, count in zip(unique, counts):
        class_name = DATASET_CONFIG['class_names'][label]
        print(f"  {class_name}: {count}")
    
    print("\n✅ Dataset and DataLoader test passed!")
