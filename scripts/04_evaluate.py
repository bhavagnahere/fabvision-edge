"""
Evaluation Script for Defect Classification Model
Generates: Confusion Matrix, Precision, Recall, F1, ROC-AUC, ECE
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATASET_CONFIG, MODELS_DIR, RESULTS_DIR, SPLITS_DIR
from scripts.dataset import create_dataloaders
from scripts.models import create_model


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(
        self,
        model: nn.Module,
        test_loader,
        device: torch.device,
        class_names: list
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Storage for predictions
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []
    
    def evaluate(self):
        """Run evaluation and collect predictions"""
        self.model.eval()
        
        print("Running evaluation...")
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                # Store results
                self.all_labels.extend(labels.cpu().numpy())
                self.all_preds.extend(preds.cpu().numpy())
                self.all_probs.extend(probs.cpu().numpy())
        
        self.all_labels = np.array(self.all_labels)
        self.all_preds = np.array(self.all_preds)
        self.all_probs = np.array(self.all_probs)
        
        print(f"✅ Evaluated {len(self.all_labels)} samples")
    
    def calculate_accuracy(self) -> float:
        """Calculate overall accuracy"""
        correct = (self.all_labels == self.all_preds).sum()
        accuracy = 100.0 * correct / len(self.all_labels)
        return accuracy
    
    def calculate_top_k_accuracy(self, k: int = 3) -> float:
        """Calculate top-k accuracy"""
        top_k_preds = np.argsort(self.all_probs, axis=1)[:, -k:]
        correct = sum([label in top_k_preds[i] for i, label in enumerate(self.all_labels)])
        top_k_acc = 100.0 * correct / len(self.all_labels)
        return top_k_acc
    
    def calculate_per_class_metrics(self) -> dict:
        """Calculate precision, recall, F1 for each class"""
        precision, recall, f1, support = precision_recall_fscore_support(
            self.all_labels,
            self.all_preds,
            average=None
        )
        
        metrics = {}
        for i, class_name in enumerate(self.class_names):
            metrics[class_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        return metrics
    
    def calculate_macro_metrics(self) -> dict:
        """Calculate macro-averaged metrics"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.all_labels,
            self.all_preds,
            average='macro'
        )
        
        return {
            'macro_precision': float(precision),
            'macro_recall': float(recall),
            'macro_f1': float(f1)
        }
    
    def calculate_weighted_metrics(self) -> dict:
        """Calculate weighted-averaged metrics"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.all_labels,
            self.all_preds,
            average='weighted'
        )
        
        return {
            'weighted_precision': float(precision),
            'weighted_recall': float(recall),
            'weighted_f1': float(f1)
        }
    
    def calculate_confusion_matrix(self) -> np.ndarray:
        """Generate confusion matrix"""
        cm = confusion_matrix(self.all_labels, self.all_preds)
        return cm
    
    def calculate_roc_auc(self) -> dict:
        """Calculate ROC-AUC scores"""
        # One-vs-rest ROC-AUC for each class
        roc_auc_scores = {}
        
        for i, class_name in enumerate(self.class_names):
            # Binary labels for this class
            binary_labels = (self.all_labels == i).astype(int)
            class_probs = self.all_probs[:, i]
            
            try:
                auc = roc_auc_score(binary_labels, class_probs)
                roc_auc_scores[class_name] = float(auc)
            except:
                roc_auc_scores[class_name] = None
        
        # Macro average
        valid_aucs = [v for v in roc_auc_scores.values() if v is not None]
        roc_auc_scores['macro_avg'] = float(np.mean(valid_aucs)) if valid_aucs else None
        
        return roc_auc_scores
    
    def calculate_ece(self, n_bins: int = 15) -> float:
        """
        Calculate Expected Calibration Error (ECE)
        Measures how well predicted probabilities match actual outcomes
        """
        confidences = np.max(self.all_probs, axis=1)
        accuracies = (self.all_preds == self.all_labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def plot_confusion_matrix(self, save_path: str):
        """Plot and save confusion matrix"""
        cm = self.calculate_confusion_matrix()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('Actual Class', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved to {save_path}")
    
    def plot_per_class_metrics(self, save_path: str):
        """Plot per-class precision, recall, F1"""
        metrics = self.calculate_per_class_metrics()
        
        classes = list(metrics.keys())
        precision = [metrics[c]['precision'] for c in classes]
        recall = [metrics[c]['recall'] for c in classes]
        f1 = [metrics[c]['f1_score'] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Per-class metrics plot saved to {save_path}")
    
    def plot_roc_curves(self, save_path: str):
        """Plot ROC curves for all classes"""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, class_name in enumerate(self.class_names):
            ax = axes[i]
            
            # Binary labels for this class
            binary_labels = (self.all_labels == i).astype(int)
            class_probs = self.all_probs[:, i]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(binary_labels, class_probs)
            auc = roc_auc_score(binary_labels, class_probs)
            
            # Plot
            ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{class_name}')
            ax.legend(loc='lower right')
            ax.grid(alpha=0.3)
        
        plt.suptitle('ROC Curves for All Classes', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ ROC curves saved to {save_path}")
    
    def generate_classification_report(self) -> str:
        """Generate sklearn classification report"""
        report = classification_report(
            self.all_labels,
            self.all_preds,
            target_names=self.class_names,
            digits=3
        )
        return report
    
    def save_results(self, save_dir: str):
        """Save all evaluation results"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate all metrics
        results = {
            'overall_accuracy': self.calculate_accuracy(),
            'top_3_accuracy': self.calculate_top_k_accuracy(3),
            'per_class_metrics': self.calculate_per_class_metrics(),
            'macro_metrics': self.calculate_macro_metrics(),
            'weighted_metrics': self.calculate_weighted_metrics(),
            'roc_auc_scores': self.calculate_roc_auc(),
            'ece': self.calculate_ece(),
            'confusion_matrix': self.calculate_confusion_matrix().tolist(),
            'classification_report': self.generate_classification_report()
        }
        
        # Save JSON results
        results_file = save_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            # Separate classification report for formatting
            report = results.pop('classification_report')
            json.dump(results, f, indent=2)
            results['classification_report'] = report
        
        # Save text report
        report_file = save_dir / 'classification_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SEMICONDUCTOR DEFECT CLASSIFICATION - EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Overall Accuracy: {results['overall_accuracy']:.2f}%\n")
            f.write(f"Top-3 Accuracy: {results['top_3_accuracy']:.2f}%\n\n")
            
            f.write("Macro-Averaged Metrics:\n")
            f.write(f"  Precision: {results['macro_metrics']['macro_precision']:.4f}\n")
            f.write(f"  Recall: {results['macro_metrics']['macro_recall']:.4f}\n")
            f.write(f"  F1-Score: {results['macro_metrics']['macro_f1']:.4f}\n\n")
            
            f.write("Weighted-Averaged Metrics:\n")
            f.write(f"  Precision: {results['weighted_metrics']['weighted_precision']:.4f}\n")
            f.write(f"  Recall: {results['weighted_metrics']['weighted_recall']:.4f}\n")
            f.write(f"  F1-Score: {results['weighted_metrics']['weighted_f1']:.4f}\n\n")
            
            f.write(f"Expected Calibration Error (ECE): {results['ece']:.4f}\n\n")
            
            f.write("="*60 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(results['classification_report'])
        
        # Plot visualizations
        self.plot_confusion_matrix(str(save_dir / 'confusion_matrix.png'))
        self.plot_per_class_metrics(str(save_dir / 'per_class_metrics.png'))
        self.plot_roc_curves(str(save_dir / 'roc_curves.png'))
        
        print(f"\n✅ All results saved to {save_dir}")
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"Macro F1-Score: {results['macro_metrics']['macro_f1']:.4f}")
        print(f"Weighted F1-Score: {results['weighted_metrics']['weighted_f1']:.4f}")
        print(f"Expected Calibration Error: {results['ece']:.4f}")
        print(f"{'='*60}\n")
        
        return results


def main():
    """Main evaluation function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load test dataloader
    print("Loading test dataset...")
    _, _, test_loader = create_dataloaders(SPLITS_DIR)
    
    # Load model
    print("Loading trained model...")
    model = create_model(
        model_type='efficientnet',
        num_classes=DATASET_CONFIG['num_classes'],
        pretrained=False
    )
    
    # Load checkpoint
    checkpoint_path = Path(MODELS_DIR) / 'checkpoint_best.pth'
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please train the model first using 03_train.py")
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"   Best validation accuracy: {checkpoint['best_val_acc']:.2f}%\n")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=DATASET_CONFIG['class_names']
    )
    
    # Run evaluation
    evaluator.evaluate()
    
    # Save results
    results_dir = Path(RESULTS_DIR) / 'evaluation'
    evaluator.save_results(results_dir)
    
    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
