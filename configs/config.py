"""
Configuration file for FabVision Edge training
"""

import os

# Dataset Configuration
DATASET_CONFIG = {
    'num_classes': 8,
    'class_names': [
        'clean',
        'particle',
        'scratch',
        'pattern_defect',
        'open_circuit',
        'short_circuit',
        'via_defect',
        'other'
    ],
    'image_size': (224, 224),
    'channels': 1,  # Grayscale
    'train_split': 0.70,
    'val_split': 0.15,
    'test_split': 0.15,
    'min_images': 500,
    'target_images': 1250,
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'train': {
        'enabled': True,
        'gaussian_noise': {'var_limit': (10.0, 50.0), 'p': 0.5},
        'elastic_transform': {'alpha': 120, 'sigma': 6, 'p': 0.3},
        'motion_blur': {'blur_limit': 7, 'p': 0.2},
        'rotate90': {'p': 0.5},
        'horizontal_flip': {'p': 0.5},
        'vertical_flip': {'p': 0.5},
        'brightness_contrast': {'brightness_limit': 0.2, 'contrast_limit': 0.2, 'p': 0.3},
    },
    'val': {
        'enabled': False,
    },
    'test': {
        'enabled': False,
    }
}

# Model Configuration
MODEL_CONFIG = {
    'architecture': 'efficientnet_lite3',
    'pretrained': True,
    'num_classes': 8,
    'dropout': 0.3,
    'attention': True,
}

# Stage 1 Binary Classifier (for cascade)
STAGE1_CONFIG = {
    'architecture': 'mobilenet_v3_small',
    'pretrained': True,
    'num_classes': 2,  # Clean vs Defect
    'input_size': (128, 128),
    'dropout': 0.2,
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 0.01,
    'optimizer': 'adamw',
    'scheduler': 'cosine',
    'early_stopping_patience': 15,
    'gradient_clip': 1.0,
    'mixed_precision': True,
    
    # Loss function
    'loss_type': 'focal',  # 'focal' or 'cross_entropy'
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'label_smoothing': 0.1,
    
    # Progressive unfreezing
    'freeze_backbone_epochs': 10,
    'unfreeze_late_blocks_epoch': 30,
}

# Quantization Configuration
QUANTIZATION_CONFIG = {
    'enabled': True,
    'method': 'qat',  # 'qat' (quantization-aware training) or 'ptq' (post-training quantization)
    'dtype': 'int8',
    'calibration_samples': 100,
}

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
SPLITS_DIR = os.path.join(DATA_DIR, 'splits')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
