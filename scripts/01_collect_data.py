"""
Data Collection Script for Semiconductor Defect Images
Downloads and organizes images from multiple public sources
"""

import os
import sys
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
from typing import List, Tuple
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import RAW_DATA_DIR, DATASET_CONFIG

class DatasetCollector:
    """Collect and organize semiconductor defect images from public sources"""
    
    def __init__(self, output_dir: str = RAW_DATA_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class directories
        for class_name in DATASET_CONFIG['class_names']:
            (self.output_dir / class_name).mkdir(exist_ok=True)
        
        self.stats = {class_name: 0 for class_name in DATASET_CONFIG['class_names']}
    
    def download_kaggle_dataset(self):
        """
        Instructions for downloading Kaggle datasets
        """
        print("\n=== KAGGLE DATASET INSTRUCTIONS ===")
        print("1. Go to https://www.kaggle.com/datasets/arnavr10880/mixedwm38")
        print("2. Download the MixedWM38 dataset")
        print("3. Extract to data/raw/kaggle_mixedwm38/")
        print("4. Run this script again")
        print("\nOR use Kaggle API:")
        print("  pip install kaggle")
        print("  kaggle datasets download -d arnavr10880/mixedwm38 -p data/raw/")
        print("  unzip data/raw/mixedwm38.zip -d data/raw/kaggle_mixedwm38/")
    
    def generate_synthetic_defects(self, num_images: int = 400):
        """
        Generate synthetic defect images using procedural generation
        """
        print(f"\n=== Generating {num_images} Synthetic Defect Images ===")
        
        image_size = DATASET_CONFIG['image_size']
        
        # Define defect generation functions
        def generate_clean(size):
            """Generate clean wafer image"""
            img = np.random.randint(200, 240, size, dtype=np.uint8)
            # Add subtle texture
            noise = np.random.normal(0, 3, size)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            return img
        
        def generate_particle(size):
            """Generate particle defects (small bright spots)"""
            img = generate_clean(size)
            num_particles = np.random.randint(1, 5)
            for _ in range(num_particles):
                x, y = np.random.randint(20, size[0]-20, 2)
                radius = np.random.randint(2, 8)
                cv2.circle(img, (x, y), radius, 255, -1)
                # Add blur
                img = cv2.GaussianBlur(img, (5, 5), 0)
            return img
        
        def generate_scratch(size):
            """Generate scratch defects (directional lines)"""
            img = generate_clean(size)
            num_scratches = np.random.randint(1, 3)
            for _ in range(num_scratches):
                angle = np.random.uniform(0, 180)
                length = np.random.randint(50, 150)
                thickness = np.random.randint(1, 4)
                
                # Random start point
                x1 = np.random.randint(0, size[0])
                y1 = np.random.randint(0, size[1])
                
                # Calculate end point
                x2 = int(x1 + length * np.cos(np.radians(angle)))
                y2 = int(y1 + length * np.sin(np.radians(angle)))
                
                cv2.line(img, (x1, y1), (x2, y2), 0, thickness)
            return img
        
        def generate_pattern_defect(size):
            """Generate pattern defects (irregular shapes)"""
            img = generate_clean(size)
            # Create grid pattern
            grid_size = 20
            for i in range(0, size[0], grid_size):
                for j in range(0, size[1], grid_size):
                    if np.random.random() < 0.3:  # 30% chance of defect in cell
                        cv2.rectangle(img, (i, j), (i+grid_size-5, j+grid_size-5), 50, -1)
            return img
        
        def generate_open_circuit(size):
            """Generate open circuit defects (broken lines)"""
            img = generate_clean(size)
            # Draw circuit-like pattern with breaks
            for _ in range(3):
                x = np.random.randint(10, size[0]-10)
                y_start = np.random.randint(10, size[1]//2)
                y_end = np.random.randint(size[1]//2, size[1]-10)
                
                # Draw line with break
                break_point = (y_start + y_end) // 2
                cv2.line(img, (x, y_start), (x, break_point-10), 100, 2)
                cv2.line(img, (x, break_point+10), (x, y_end), 100, 2)
            return img
        
        def generate_short_circuit(size):
            """Generate short circuit defects (unintended connections)"""
            img = generate_clean(size)
            # Draw parallel lines with bridge
            y1, y2 = size[1]//3, 2*size[1]//3
            for x in range(20, size[0]-20, 40):
                cv2.line(img, (x, y1), (x+30, y1), 80, 2)
                cv2.line(img, (x, y2), (x+30, y2), 80, 2)
                # Add short
                if np.random.random() < 0.5:
                    bridge_x = x + 15
                    cv2.line(img, (bridge_x, y1), (bridge_x, y2), 80, 2)
            return img
        
        def generate_via_defect(size):
            """Generate via defects (malformed holes)"""
            img = generate_clean(size)
            num_vias = np.random.randint(10, 20)
            for _ in range(num_vias):
                x = np.random.randint(30, size[0]-30)
                y = np.random.randint(30, size[1]-30)
                # Some vias are defective (irregular shape or missing)
                if np.random.random() < 0.4:
                    # Irregular via
                    pts = []
                    for angle in range(0, 360, 60):
                        r = np.random.randint(3, 8)
                        px = int(x + r * np.cos(np.radians(angle)))
                        py = int(y + r * np.sin(np.radians(angle)))
                        pts.append([px, py])
                    cv2.fillPoly(img, [np.array(pts)], 0)
                else:
                    # Normal via
                    cv2.circle(img, (x, y), 5, 0, -1)
            return img
        
        def generate_other(size):
            """Generate other defects (pitting, corrosion, edge defects)"""
            img = generate_clean(size)
            defect_type = np.random.choice(['pitting', 'corrosion', 'edge'])
            
            if defect_type == 'pitting':
                num_pits = np.random.randint(20, 50)
                for _ in range(num_pits):
                    x = np.random.randint(0, size[0])
                    y = np.random.randint(0, size[1])
                    cv2.circle(img, (x, y), 2, 150, -1)
            
            elif defect_type == 'corrosion':
                # Random corrosion pattern
                mask = np.random.random(size) < 0.1
                img[mask] = np.random.randint(100, 150, np.sum(mask))
            
            else:  # edge defect
                # Damage at edges
                edge_size = 20
                img[:edge_size, :] = np.random.randint(50, 100, (edge_size, size[1]))
            
            return img
        
        # Generation mapping
        generators = {
            'clean': generate_clean,
            'particle': generate_particle,
            'scratch': generate_scratch,
            'pattern_defect': generate_pattern_defect,
            'open_circuit': generate_open_circuit,
            'short_circuit': generate_short_circuit,
            'via_defect': generate_via_defect,
            'other': generate_other,
        }
        
        # Generate images for each class
        images_per_class = num_images // len(DATASET_CONFIG['class_names'])
        
        for class_name in tqdm(DATASET_CONFIG['class_names'], desc="Generating classes"):
            generator = generators[class_name]
            class_dir = self.output_dir / class_name
            
            for i in range(images_per_class):
                img = generator(image_size)
                
                # Apply CLAHE for better contrast
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)
                
                # Save image
                filename = f"synthetic_{class_name}_{i:04d}.png"
                cv2.imwrite(str(class_dir / filename), img)
                self.stats[class_name] += 1
    
    def organize_kaggle_data(self):
        """
        Organize downloaded Kaggle data into class folders
        """
        kaggle_dir = self.output_dir / 'kaggle_mixedwm38'
        
        if not kaggle_dir.exists():
            print(f"Kaggle directory not found: {kaggle_dir}")
            return
        
        print("\n=== Organizing Kaggle Data ===")
        
        # This is a placeholder - you'll need to adapt based on actual Kaggle dataset structure
        # The MixedWM38 dataset structure needs to be explored first
        print("Please manually organize Kaggle images into respective class folders")
        print(f"Place images in: {self.output_dir / '<class_name>'}")
    
    def validate_dataset(self) -> bool:
        """
        Validate collected dataset meets requirements
        """
        print("\n=== Dataset Validation ===")
        
        total_images = sum(self.stats.values())
        print(f"Total images collected: {total_images}")
        
        for class_name, count in self.stats.items():
            print(f"  {class_name}: {count} images")
        
        # Check minimum requirement
        if total_images < DATASET_CONFIG['min_images']:
            print(f"❌ Dataset too small! Need at least {DATASET_CONFIG['min_images']} images")
            return False
        
        # Check balance
        max_count = max(self.stats.values())
        min_count = min(self.stats.values())
        balance_ratio = min_count / max_count if max_count > 0 else 0
        
        print(f"\nDataset Balance Ratio: {balance_ratio:.2f}")
        if balance_ratio < 0.5:
            print("⚠️  Warning: Dataset is imbalanced (ratio < 0.5)")
        
        print("\n✅ Dataset validation passed!")
        return True
    
    def create_metadata(self):
        """Create metadata file for the dataset"""
        metadata = {
            'total_images': sum(self.stats.values()),
            'class_distribution': self.stats,
            'class_names': DATASET_CONFIG['class_names'],
            'image_size': DATASET_CONFIG['image_size'],
            'channels': DATASET_CONFIG['channels'],
        }
        
        metadata_file = self.output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Metadata saved to {metadata_file}")
    
    def count_existing_images(self):
        """Count images already in the dataset"""
        for class_name in DATASET_CONFIG['class_names']:
            class_dir = self.output_dir / class_name
            if class_dir.exists():
                images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
                self.stats[class_name] = len(images)

def main():
    """Main data collection pipeline"""
    print("=" * 60)
    print("FabVision Edge - Data Collection")
    print("=" * 60)
    
    collector = DatasetCollector()
    
    # Count existing images
    collector.count_existing_images()
    
    # Display Kaggle download instructions
    collector.download_kaggle_dataset()
    
    # Generate synthetic images
    response = input("\nGenerate synthetic images? (y/n): ")
    if response.lower() == 'y':
        num_synthetic = int(input("How many synthetic images to generate? (default 400): ") or 400)
        collector.generate_synthetic_defects(num_synthetic)
    
    # Count again after generation
    collector.count_existing_images()
    
    # Validate dataset
    if collector.validate_dataset():
        collector.create_metadata()
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print(f"Images saved to: {RAW_DATA_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
