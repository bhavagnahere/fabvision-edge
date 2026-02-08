"""
Data Preprocessing and Train/Val/Test Splitting
Applies quality checks, preprocessing, and creates dataset splits
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import json
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import hashlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR,
    DATASET_CONFIG, AUGMENTATION_CONFIG
)

class ImagePreprocessor:
    """Preprocess and quality-check semiconductor images"""
    
    def __init__(self):
        self.image_size = DATASET_CONFIG['image_size']
        self.rejected_count = 0
        self.processed_count = 0
        self.duplicate_hashes = set()
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate Laplacian variance (sharpness metric)"""
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return laplacian.var()
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate contrast ratio"""
        min_val = image.min()
        max_val = image.max()
        if min_val == 0:
            return float('inf')
        return max_val / min_val
    
    def calculate_snr(self, image: np.ndarray) -> float:
        """Calculate signal-to-noise ratio (approximation)"""
        # Use standard deviation as a proxy for noise
        mean = image.mean()
        std = image.std()
        if std == 0:
            return float('inf')
        # SNR in dB
        snr_db = 20 * np.log10(mean / std) if std > 0 else 0
        return snr_db
    
    def calculate_perceptual_hash(self, image: np.ndarray) -> str:
        """Calculate perceptual hash for duplicate detection"""
        # Resize to 8x8
        small = cv2.resize(image, (8, 8))
        # Calculate mean
        avg = small.mean()
        # Create binary hash
        diff = small > avg
        # Convert to hex string
        hash_str = ''.join(['1' if pixel else '0' for pixel in diff.flatten()])
        return hash_str
    
    def quality_check(self, image: np.ndarray, filepath: str) -> Tuple[bool, str]:
        """
        Perform quality checks on image
        Returns: (is_valid, rejection_reason)
        """
        # Check sharpness
        sharpness = self.calculate_sharpness(image)
        if sharpness < 80:
            return False, f"Blurry (sharpness={sharpness:.1f})"
        
        # Check contrast
        contrast = self.calculate_contrast(image)
        if contrast < 2.5 and contrast != float('inf'):
            return False, f"Low contrast (ratio={contrast:.2f})"
        
        # Check SNR
        snr = self.calculate_snr(image)
        if snr < 35:
            return False, f"Too noisy (SNR={snr:.1f}dB)"
        
        # Check for duplicates
        img_hash = self.calculate_perceptual_hash(image)
        if img_hash in self.duplicate_hashes:
            return False, "Duplicate detected"
        self.duplicate_hashes.add(img_hash)
        
        return True, "OK"
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Complete preprocessing pipeline
        """
        # 1. Load as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # 2. Resize with antialiasing
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        
        # 3. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        return img
    
    def process_dataset(self, input_dir: Path, output_dir: Path):
        """Process all images in the dataset"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create quality report
        quality_report = []
        
        for class_name in DATASET_CONFIG['class_names']:
            class_input_dir = input_dir / class_name
            class_output_dir = output_dir / class_name
            class_output_dir.mkdir(exist_ok=True)
            
            if not class_input_dir.exists():
                print(f"Warning: {class_input_dir} does not exist")
                continue
            
            # Get all images
            image_files = list(class_input_dir.glob('*.png')) + \
                         list(class_input_dir.glob('*.jpg')) + \
                         list(class_input_dir.glob('*.jpeg'))
            
            print(f"\nProcessing {class_name}: {len(image_files)} images")
            
            for img_path in tqdm(image_files, desc=f"  {class_name}"):
                try:
                    # Preprocess
                    img = self.preprocess_image(str(img_path))
                    
                    # Quality check
                    is_valid, reason = self.quality_check(img, str(img_path))
                    
                    quality_report.append({
                        'file': str(img_path),
                        'class': class_name,
                        'valid': is_valid,
                        'reason': reason
                    })
                    
                    if is_valid:
                        # Save processed image
                        output_path = class_output_dir / img_path.name
                        cv2.imwrite(str(output_path), img)
                        self.processed_count += 1
                    else:
                        self.rejected_count += 1
                
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    quality_report.append({
                        'file': str(img_path),
                        'class': class_name,
                        'valid': False,
                        'reason': str(e)
                    })
                    self.rejected_count += 1
        
        # Save quality report
        report_path = output_dir / 'quality_report.json'
        with open(report_path, 'w') as f:
            json.dump(quality_report, f, indent=2)
        
        print(f"\n✅ Processing complete!")
        print(f"  Processed: {self.processed_count} images")
        print(f"  Rejected: {self.rejected_count} images")
        print(f"  Quality report: {report_path}")


class DatasetSplitter:
    """Split dataset into train/val/test sets"""
    
    def __init__(self):
        self.train_split = DATASET_CONFIG['train_split']
        self.val_split = DATASET_CONFIG['val_split']
        self.test_split = DATASET_CONFIG['test_split']
    
    def split_dataset(self, input_dir: Path, output_dir: Path):
        """
        Split dataset maintaining class balance
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            for class_name in DATASET_CONFIG['class_names']:
                (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        split_stats = {
            'train': {class_name: 0 for class_name in DATASET_CONFIG['class_names']},
            'val': {class_name: 0 for class_name in DATASET_CONFIG['class_names']},
            'test': {class_name: 0 for class_name in DATASET_CONFIG['class_names']},
        }
        
        for class_name in DATASET_CONFIG['class_names']:
            class_dir = input_dir / class_name
            
            if not class_dir.exists():
                continue
            
            # Get all images for this class
            images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
            
            if len(images) == 0:
                print(f"Warning: No images found for {class_name}")
                continue
            
            # Split into train+val and test first
            train_val_images, test_images = train_test_split(
                images,
                test_size=self.test_split,
                random_state=42
            )
            
            # Split train+val into train and val
            val_size = self.val_split / (self.train_split + self.val_split)
            train_images, val_images = train_test_split(
                train_val_images,
                test_size=val_size,
                random_state=42
            )
            
            # Copy files to respective directories
            for img_path in train_images:
                dest = output_dir / 'train' / class_name / img_path.name
                shutil.copy2(img_path, dest)
                split_stats['train'][class_name] += 1
            
            for img_path in val_images:
                dest = output_dir / 'val' / class_name / img_path.name
                shutil.copy2(img_path, dest)
                split_stats['val'][class_name] += 1
            
            for img_path in test_images:
                dest = output_dir / 'test' / class_name / img_path.name
                shutil.copy2(img_path, dest)
                split_stats['test'][class_name] += 1
        
        # Print statistics
        print("\n=== Dataset Split Statistics ===")
        for split in ['train', 'val', 'test']:
            total = sum(split_stats[split].values())
            print(f"\n{split.upper()}: {total} images")
            for class_name, count in split_stats[split].items():
                print(f"  {class_name}: {count}")
        
        # Save split metadata
        metadata = {
            'splits': split_stats,
            'split_ratios': {
                'train': self.train_split,
                'val': self.val_split,
                'test': self.test_split,
            },
            'total': {
                'train': sum(split_stats['train'].values()),
                'val': sum(split_stats['val'].values()),
                'test': sum(split_stats['test'].values()),
            }
        }
        
        metadata_path = output_dir / 'split_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Split metadata saved to {metadata_path}")
        
        return split_stats


def main():
    """Main preprocessing pipeline"""
    print("=" * 60)
    print("FabVision Edge - Data Preprocessing & Splitting")
    print("=" * 60)
    
    # Step 1: Preprocess images
    print("\nStep 1: Preprocessing images...")
    preprocessor = ImagePreprocessor()
    preprocessor.process_dataset(
        input_dir=Path(RAW_DATA_DIR),
        output_dir=Path(PROCESSED_DATA_DIR)
    )
    
    # Step 2: Split dataset
    print("\nStep 2: Splitting dataset...")
    splitter = DatasetSplitter()
    split_stats = splitter.split_dataset(
        input_dir=Path(PROCESSED_DATA_DIR),
        output_dir=Path(SPLITS_DIR)
    )
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Processed data: {PROCESSED_DATA_DIR}")
    print(f"Split data: {SPLITS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
