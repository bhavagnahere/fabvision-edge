"""
Test script to verify installation and dependencies
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'albumentations': 'Albumentations',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'TQDM',
        'onnx': 'ONNX',
        'onnxruntime': 'ONNX Runtime',
    }
    
    failed = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages installed correctly!")
        return True


def test_directory_structure():
    """Test if directory structure exists"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'configs',
        'data',
        'models',
        'scripts',
        'results',
    ]
    
    missing = []
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ - creating...")
            dir_path.mkdir(parents=True, exist_ok=True)
            missing.append(dir_name)
    
    if missing:
        print(f"\n⚠️  Created missing directories: {', '.join(missing)}")
    else:
        print("\n✅ Directory structure is correct!")
    
    return True


def test_config():
    """Test if config can be loaded"""
    print("\nTesting configuration...")
    
    try:
        sys.path.append(str(Path.cwd()))
        from configs.config import DATASET_CONFIG, TRAINING_CONFIG, MODEL_CONFIG
        
        print(f"  ✓ Config loaded")
        print(f"  - Number of classes: {DATASET_CONFIG['num_classes']}")
        print(f"  - Image size: {DATASET_CONFIG['image_size']}")
        print(f"  - Batch size: {TRAINING_CONFIG['batch_size']}")
        print(f"  - Model architecture: {MODEL_CONFIG['architecture']}")
        
        print("\n✅ Configuration is valid!")
        return True
    
    except Exception as e:
        print(f"\n❌ Configuration error: {e}")
        return False


def test_model_creation():
    """Test if models can be created"""
    print("\nTesting model creation...")
    
    try:
        import torch
        sys.path.append(str(Path.cwd()))
        from scripts.models import create_model, count_parameters
        
        # Test EfficientNet creation
        model = create_model('efficientnet', num_classes=8, pretrained=False)
        params = count_parameters(model)
        
        print(f"  ✓ EfficientNet created")
        print(f"  - Parameters: {params:,}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 1, 224, 224)
        output = model(dummy_input)
        
        print(f"  ✓ Forward pass successful")
        print(f"  - Output shape: {output.shape}")
        
        print("\n✅ Model creation works correctly!")
        return True
    
    except Exception as e:
        print(f"\n❌ Model creation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cuda():
    """Test CUDA availability"""
    print("\nTesting CUDA...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"  - Device count: {torch.cuda.device_count()}")
            print(f"  - Device name: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA version: {torch.version.cuda}")
        else:
            print(f"  ⚠️  CUDA not available (will use CPU)")
            print(f"  - Training will be slower on CPU")
        
        return True
    
    except Exception as e:
        print(f"\n❌ CUDA test error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*70)
    print("FabVision Edge - Installation Test")
    print("="*70)
    
    tests = [
        ("Package Imports", test_imports),
        ("Directory Structure", test_directory_structure),
        ("Configuration", test_config),
        ("Model Creation", test_model_creation),
        ("CUDA Support", test_cuda),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
        
        print()
    
    # Summary
    print("="*70)
    print("Test Summary")
    print("="*70)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status:10s} - {test_name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 All tests passed! You're ready to go!")
        print("\nQuick Start:")
        print("  1. Generate data:     python scripts/01_collect_data.py")
        print("  2. Preprocess:        python scripts/02_preprocess_data.py")
        print("  3. Train:             python scripts/03_train.py")
        print("  4. Evaluate:          python scripts/04_evaluate.py")
        print("  5. Export:            python scripts/05_export_onnx.py")
        print("\nOr run full pipeline: python run_pipeline.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues before proceeding.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check Python version: python --version (need 3.8+)")
        print("  - Update PyTorch: pip install --upgrade torch torchvision")
    
    print("="*70)


if __name__ == "__main__":
    main()
