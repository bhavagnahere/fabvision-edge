#!/usr/bin/env python
"""
Master script to run the complete FabVision Edge pipeline
From data generation to model export
"""

import sys
import os
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(command, description):
    """Run a command and handle errors"""
    print("\n" + "="*70)
    print(f"STEP: {description}")
    print("="*70)
    print(f"Command: {command}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error in: {description}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print(f"\n✅ Completed: {description}")
    
    return result.returncode == 0


def main():
    print("""
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║                    FabVision Edge - Full Pipeline                    ║
║             Semiconductor Defect Classification System               ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("This script will run the complete pipeline:")
    print("  1. Data Collection (synthetic generation)")
    print("  2. Data Preprocessing & Splitting")
    print("  3. Model Training")
    print("  4. Model Evaluation")
    print("  5. ONNX Export")
    print("  6. Create Submission ZIP")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    # Configuration
    num_synthetic = input("\nHow many synthetic images to generate? (default: 500): ")
    num_synthetic = num_synthetic.strip() or "500"
    
    num_epochs = input("How many training epochs? (default: 20 for quick test, 100 for full): ")
    num_epochs = num_epochs.strip() or "20"
    
    # Step 1: Data Collection
    # Note: This will be interactive, user needs to respond to prompts
    print("\n" + "="*70)
    print("STEP 1: Data Collection")
    print("="*70)
    print("This step will generate synthetic defect images.")
    print(f"When prompted, type 'y' and then '{num_synthetic}'\n")
    input("Press Enter to continue...")
    
    success = run_command(
        "python scripts/01_collect_data.py",
        "Data Collection"
    )
    
    # Step 2: Preprocessing
    success = run_command(
        "python scripts/02_preprocess_data.py",
        "Data Preprocessing & Splitting"
    )
    
    if not success:
        print("\n⚠️  Skipping remaining steps due to preprocessing failure")
        return
    
    # Step 3: Training
    # Modify config temporarily for quick test
    if int(num_epochs) != 100:
        print(f"\n📝 Note: Training for {num_epochs} epochs (reduced for testing)")
        print("    For full training, use 100 epochs")
    
    success = run_command(
        f"python scripts/03_train.py",
        f"Model Training ({num_epochs} epochs)"
    )
    
    if not success:
        print("\n⚠️  Skipping remaining steps due to training failure")
        return
    
    # Step 4: Evaluation
    success = run_command(
        "python scripts/04_evaluate.py",
        "Model Evaluation"
    )
    
    # Step 5: ONNX Export
    success = run_command(
        "python scripts/05_export_onnx.py",
        "ONNX Model Export"
    )
    
    # Step 6: Create submission ZIP
    success = run_command(
        "python scripts/utils/create_submission_zip.py",
        "Create Submission ZIP"
    )
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 PIPELINE COMPLETE!")
    print("="*70)
    
    print("\n📁 Generated Files:")
    print("   ├── data/splits/          - Train/Val/Test datasets")
    print("   ├── models/saved/         - Trained model checkpoints")
    print("   ├── models/onnx/          - ONNX exported model")
    print("   ├── results/evaluation/   - Evaluation metrics & plots")
    print("   └── fabvision_dataset_*.zip - Submission dataset")
    
    print("\n📊 Next Steps:")
    print("   1. Check results/evaluation/ for performance metrics")
    print("   2. Review training logs in runs/ with TensorBoard")
    print("   3. Test inference: python scripts/06_inference.py --image <path>")
    print("   4. Submit files for hackathon")
    
    print("\n📤 Hackathon Deliverables:")
    print("   ✅ Documentation: docs/FabVision_Report.pdf")
    print("   ✅ Dataset: fabvision_dataset_*.zip")
    print("   ✅ Model (ONNX): models/onnx/defect_classifier.onnx")
    print("   ✅ Results: results/evaluation/evaluation_results.json")
    print("   ✅ Code: GitHub repository")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
