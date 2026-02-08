"""
Utility script to package dataset for hackathon submission
Creates properly structured ZIP file
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import SPLITS_DIR, DATASET_CONFIG


def create_dataset_zip(output_name: str = None):
    """
    Create ZIP file with dataset in required format:
    
    dataset.zip/
    ├── train/
    │   ├── clean/
    │   ├── particle/
    │   └── ...
    ├── val/
    │   ├── clean/
    │   └── ...
    └── test/
        ├── clean/
        └── ...
    """
    
    if output_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_name = f'fabvision_dataset_{timestamp}.zip'
    
    splits_dir = Path(SPLITS_DIR)
    
    if not splits_dir.exists():
        print(f"❌ Dataset not found at {splits_dir}")
        print("Please run 02_preprocess_data.py first")
        return None
    
    print("="*60)
    print("Creating Dataset ZIP for Hackathon Submission")
    print("="*60)
    
    # Check dataset completeness
    print("\n📊 Checking dataset...")
    stats = {'train': {}, 'val': {}, 'test': {}}
    total_images = 0
    
    for split in ['train', 'val', 'test']:
        split_dir = splits_dir / split
        for class_name in DATASET_CONFIG['class_names']:
            class_dir = split_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg')))
                stats[split][class_name] = count
                total_images += count
            else:
                stats[split][class_name] = 0
    
    print(f"\nTotal images: {total_images}")
    for split in ['train', 'val', 'test']:
        split_total = sum(stats[split].values())
        print(f"  {split}: {split_total} images")
    
    if total_images < 500:
        print(f"\n⚠️  Warning: Dataset has only {total_images} images (minimum: 500)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return None
    
    # Create ZIP file
    print(f"\n📦 Creating ZIP file: {output_name}")
    
    with zipfile.ZipFile(output_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for split in ['train', 'val', 'test']:
            split_dir = splits_dir / split
            
            for class_name in DATASET_CONFIG['class_names']:
                class_dir = split_dir / class_name
                
                if not class_dir.exists():
                    continue
                
                # Add all images from this class
                images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
                
                for img_path in images:
                    # Archive path: split/class_name/image.png
                    arcname = f"{split}/{class_name}/{img_path.name}"
                    zipf.write(img_path, arcname)
        
        # Add metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'total_images': total_images,
            'splits': stats,
            'class_names': DATASET_CONFIG['class_names'],
            'num_classes': DATASET_CONFIG['num_classes'],
            'image_size': DATASET_CONFIG['image_size'],
        }
        
        metadata_str = json.dumps(metadata, indent=2)
        zipf.writestr('metadata.json', metadata_str)
        
        # Add README
        readme = f"""FabVision Edge - Semiconductor Defect Dataset
================================================

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Images: {total_images}

Dataset Structure:
------------------
train/           - Training set ({sum(stats['train'].values())} images)
val/             - Validation set ({sum(stats['val'].values())} images)
test/            - Test set ({sum(stats['test'].values())} images)

Classes:
--------
{chr(10).join(f'- {name}' for name in DATASET_CONFIG['class_names'])}

Class Distribution:
-------------------
TRAIN:
{chr(10).join(f'  {name}: {stats["train"][name]}' for name in DATASET_CONFIG['class_names'])}

VAL:
{chr(10).join(f'  {name}: {stats["val"][name]}' for name in DATASET_CONFIG['class_names'])}

TEST:
{chr(10).join(f'  {name}: {stats["test"][name]}' for name in DATASET_CONFIG['class_names'])}

Image Specifications:
---------------------
- Format: PNG/JPG
- Color: Grayscale (1 channel)
- Size: {DATASET_CONFIG['image_size'][0]}x{DATASET_CONFIG['image_size'][1]} pixels
- Preprocessing: CLAHE applied, quality-checked

Usage:
------
Extract the ZIP file and load images using your preferred framework.
All images are preprocessed and ready for training/inference.

Contact:
--------
GitHub: https://github.com/yourusername/fabvision-edge
"""
        zipf.writestr('README.txt', readme)
    
    # Get file size
    file_size_mb = Path(output_name).stat().st_size / (1024 * 1024)
    
    print(f"\n✅ Dataset ZIP created successfully!")
    print(f"   File: {output_name}")
    print(f"   Size: {file_size_mb:.2f} MB")
    print(f"   Images: {total_images}")
    
    return output_name


def verify_zip(zip_path: str):
    """Verify ZIP file structure"""
    print(f"\n🔍 Verifying {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        files = zipf.namelist()
        
        print(f"   Total files: {len(files)}")
        
        # Check for required structure
        has_train = any('train/' in f for f in files)
        has_val = any('val/' in f for f in files)
        has_test = any('test/' in f for f in files)
        has_metadata = 'metadata.json' in files
        
        print(f"   ✓ train/ folder" if has_train else "   ✗ train/ folder missing")
        print(f"   ✓ val/ folder" if has_val else "   ✗ val/ folder missing")
        print(f"   ✓ test/ folder" if has_test else "   ✗ test/ folder missing")
        print(f"   ✓ metadata.json" if has_metadata else "   ✗ metadata.json missing")
        
        if all([has_train, has_val, has_test, has_metadata]):
            print("\n✅ ZIP structure is valid!")
            return True
        else:
            print("\n❌ ZIP structure is invalid!")
            return False


def main():
    print("="*60)
    print("FabVision Edge - Dataset Packaging Tool")
    print("="*60)
    
    # Create ZIP
    zip_path = create_dataset_zip()
    
    if zip_path is None:
        print("\n❌ Dataset packaging cancelled")
        return
    
    # Verify ZIP
    verify_zip(zip_path)
    
    print("\n" + "="*60)
    print("✅ Ready for Hackathon Submission!")
    print("="*60)
    print(f"\nSubmit this file: {zip_path}")
    print("\nNext steps:")
    print("1. Upload dataset ZIP to hackathon portal")
    print("2. Train model: python scripts/03_train.py")
    print("3. Evaluate: python scripts/04_evaluate.py")
    print("4. Export ONNX: python scripts/05_export_onnx.py")
    print("5. Submit model + results")


if __name__ == "__main__":
    main()
