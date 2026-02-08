# FabVision Edge - Setup & Usage Guide

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Detailed Setup](#detailed-setup)
3. [Pipeline Execution](#pipeline-execution)
4. [Hackathon Deliverables](#hackathon-deliverables)
5. [Troubleshooting](#troubleshooting)

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Install Dependencies
```bash
cd fabvision-edge
pip install -r requirements.txt
```

### Step 2: Test Installation
```bash
python test_installation.py
```

### Step 3: Run Full Pipeline
```bash
python run_pipeline.py
```

This will:
- Generate 500 synthetic defect images
- Preprocess and split data (70/15/15)
- Train the model
- Evaluate performance
- Export to ONNX
- Create submission ZIP

**Time: ~30 minutes** (depends on GPU)

---

## 🔧 Detailed Setup

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU training)
- 8GB RAM minimum
- 10GB free disk space

### Installation Steps

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install PyTorch (choose based on your CUDA version)
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Verify installation
python test_installation.py
```

---

## 📦 Pipeline Execution

### Option A: Automated Pipeline (Recommended)

```bash
python run_pipeline.py
```

When prompted:
- Number of synthetic images: **500** (minimum) or **1000** (recommended)
- Training epochs: **20** (quick test) or **100** (full training)

### Option B: Manual Step-by-Step

#### 1. Data Collection
```bash
python scripts/01_collect_data.py
```

**What it does:**
- Generates synthetic defect images
- Creates 8 defect classes + clean class
- Saves to `data/raw/`

**Options:**
- Use synthetic generation (easiest)
- Download Kaggle datasets (optional)
- Use your own images (organize into class folders)

#### 2. Preprocessing
```bash
python scripts/02_preprocess_data.py
```

**What it does:**
- Quality checks (sharpness, contrast, SNR)
- Applies CLAHE enhancement
- Removes duplicates
- Splits into train/val/test (70/15/15)

**Output:** `data/splits/`

#### 3. Training
```bash
python scripts/03_train.py
```

**What it does:**
- Loads preprocessed data
- Trains EfficientNet-Lite3 with attention
- Uses focal loss + label smoothing
- Applies progressive unfreezing
- Saves best checkpoint

**Output:** `models/saved/checkpoint_best.pth`

**Monitor training:**
```bash
tensorboard --logdir runs/
```
Then open http://localhost:6006

#### 4. Evaluation
```bash
python scripts/04_evaluate.py
```

**What it does:**
- Loads best model
- Evaluates on test set
- Generates confusion matrix
- Calculates all metrics

**Output:** `results/evaluation/`
- evaluation_results.json
- confusion_matrix.png
- per_class_metrics.png
- roc_curves.png

#### 5. ONNX Export
```bash
python scripts/05_export_onnx.py
```

**What it does:**
- Converts PyTorch → ONNX
- Validates conversion
- Benchmarks inference speed

**Output:** `models/onnx/defect_classifier.onnx`

#### 6. Create Submission ZIP
```bash
python scripts/utils/create_submission_zip.py
```

**What it does:**
- Packages dataset in required format
- Includes metadata
- Creates `fabvision_dataset_YYYYMMDD_HHMMSS.zip`

---

## 📋 Hackathon Deliverables Checklist

### Phase 1 Deliverables

- [ ] **Document (PDF)**
  - Location: `docs/FabVision_Report.pdf`
  - Already created from your LaTeX code
  - Describes methodology, dataset, results

- [ ] **Dataset (ZIP)**
  - Created by: `scripts/utils/create_submission_zip.py`
  - Format: train/val/test folders with class subfolders
  - Minimum 500 images

- [ ] **Trained Model (ONNX)**
  - Location: `models/onnx/defect_classifier.onnx`
  - Created by: `scripts/05_export_onnx.py`
  - Ready for NXP eIQ deployment

- [ ] **Model Results**
  - Location: `results/evaluation/evaluation_results.json`
  - Includes: Accuracy, Precision, Recall, F1, Confusion Matrix
  - Model size reported

- [ ] **GitHub Repository**
  - Complete codebase
  - README with instructions
  - requirements.txt
  - All scripts organized

### Submission Files Summary

```
hackathon_submission/
├── FabVision_Report.pdf              # Technical documentation
├── fabvision_dataset_*.zip           # Dataset (500-1250 images)
├── defect_classifier.onnx            # Trained model
├── evaluation_results.json           # Performance metrics
└── github_repository_link.txt        # Link to code repo
```

---

## 🧪 Testing Inference

### Single Image
```bash
python scripts/06_inference.py --image path/to/test_image.png
```

### Directory of Images
```bash
python scripts/06_inference.py --image path/to/test_images/
```

### With Custom Model
```bash
python scripts/06_inference.py \
  --image test.png \
  --model models/saved/checkpoint_best.pth \
  --top-k 5
```

---

## 🎯 Expected Results

After running the full pipeline, you should achieve:

| Metric | Target Value |
|--------|--------------|
| Overall Accuracy | 94-97% |
| Macro F1-Score | >0.90 |
| Model Size (FP32) | ~17 MB |
| Model Size (ONNX) | ~17 MB |
| Inference Time (CPU) | <50 ms |
| Total Images | 500-1250 |

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
- Reduce batch size in `configs/config.py`:
  ```python
  TRAINING_CONFIG = {
      'batch_size': 16,  # Reduce from 32
      ...
  }
  ```
- Or train on CPU (slower but works)

### Issue: "Module not found"
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

### Issue: "Not enough images"
**Solution:**
- Generate more synthetic images in step 1
- Or download additional datasets from Kaggle

### Issue: Model not converging
**Solution:**
- Check data quality in `data/processed/quality_report.json`
- Increase training epochs
- Adjust learning rate in `configs/config.py`

### Issue: Poor accuracy on specific class
**Solution:**
- Check class balance in dataset
- Generate more samples for that class
- Review `results/evaluation/confusion_matrix.png`

---

## 📊 Performance Optimization

### For Faster Training:
1. Use GPU (10-20× speedup)
2. Increase batch size (if memory allows)
3. Reduce number of epochs for testing
4. Enable mixed precision (already enabled)

### For Better Accuracy:
1. Generate more training data (1000+ images)
2. Train for full 100 epochs
3. Use data augmentation (already enabled)
4. Fine-tune hyperparameters

### For Smaller Model:
1. Apply quantization (Phase 3)
2. Use knowledge distillation
3. Prune unnecessary weights
4. Convert to TensorFlow Lite

---

## 🔄 Iterative Development

### Experiment Loop:
1. **Hypothesis:** "More augmentation improves accuracy"
2. **Modify:** Edit `configs/config.py` augmentation settings
3. **Train:** `python scripts/03_train.py`
4. **Evaluate:** Check results in `results/evaluation/`
5. **Compare:** Look at confusion matrix, per-class metrics
6. **Repeat:** Iterate until satisfied

### Track Experiments:
- Use TensorBoard for training curves
- Save results with timestamps
- Document changes in a experiment log

---

## 📚 Additional Resources

### Understanding the Code:
- `configs/config.py` - All hyperparameters
- `scripts/models.py` - Model architectures
- `scripts/dataset.py` - Data loading & augmentation
- `scripts/03_train.py` - Training loop

### Modifying Architecture:
To use a different backbone:
```python
# In configs/config.py
MODEL_CONFIG = {
    'architecture': 'mobilenet',  # or 'efficientnet'
    ...
}
```

### Adding New Defect Class:
1. Add to `DATASET_CONFIG['class_names']` in config.py
2. Create folder in `data/raw/`
3. Add images to folder
4. Re-run preprocessing

---

## 🎓 Learning Path

### For Beginners:
1. Start with provided synthetic data
2. Train for 20 epochs (quick)
3. Understand evaluation metrics
4. Try modifying hyperparameters

### For Intermediate:
1. Download real datasets
2. Train for full 100 epochs
3. Implement cascade architecture
4. Export and benchmark ONNX

### For Advanced:
1. Implement quantization-aware training
2. Deploy on actual edge device
3. Optimize for specific hardware
4. Contribute improvements

---

## 💡 Tips for Hackathon Success

1. **Start Simple:** Get baseline working first
2. **Document Everything:** Keep notes on what works
3. **Visualize Results:** Use plots to understand model
4. **Test Early:** Don't wait until end to test submission format
5. **Ask Questions:** Use GitHub issues for help

---

## 📞 Getting Help

If you encounter issues:
1. Check this troubleshooting guide
2. Run `python test_installation.py`
3. Check GitHub issues
4. Review error messages carefully
5. Check TensorBoard for training curves

---

## ✅ Pre-Submission Checklist

Before submitting to hackathon:

- [ ] Dataset has 500+ images
- [ ] All 8 classes present
- [ ] Model accuracy > 90%
- [ ] ONNX export successful
- [ ] ZIP file structure correct
- [ ] README is clear
- [ ] GitHub repo is public
- [ ] All code commented
- [ ] Results documented

---

**Good luck with the hackathon! 🚀**

Remember: The goal is to demonstrate understanding of edge AI concepts, not just achieve the highest accuracy. Focus on:
- Clean, well-documented code
- Understanding of the problem
- Thoughtful architecture choices
- Comprehensive evaluation
- Deployment readiness
