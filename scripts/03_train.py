"""
Training Script for Defect Classification Model
Includes: Focal Loss, Label Smoothing, Mixed Precision, Progressive Unfreezing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    TRAINING_CONFIG, MODEL_CONFIG, SPLITS_DIR,
    MODELS_DIR, RESULTS_DIR, DATASET_CONFIG
)
from scripts.dataset import create_dataloaders
from scripts.models import create_model, count_parameters, get_model_size_mb


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert to probabilities
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy with Label Smoothing"""
    
    def __init__(self, epsilon: float = 0.1, reduction: str = 'mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = inputs.size(-1)
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        
        # Convert targets to one-hot
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / n_classes
        
        # Calculate loss
        loss = -(targets_smooth * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLossWithLabelSmoothing(nn.Module):
    """Combined Focal Loss and Label Smoothing"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, epsilon: float = 0.1):
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = inputs.size(-1)
        
        # Label smoothing
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = (1 - self.epsilon) * targets_one_hot + self.epsilon / n_classes
        
        # Focal loss with smooth labels
        log_probs = nn.functional.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        focal_weight = self.alpha * (1 - probs) ** self.gamma
        loss = -(focal_weight * targets_smooth * log_probs).sum(dim=-1)
        
        return loss.mean()


class Trainer:
    """Training manager for defect classification"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: torch.device,
        config: Dict = TRAINING_CONFIG,
        save_dir: str = MODELS_DIR
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup loss function
        self.criterion = self._setup_loss()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Mixed precision
        self.use_amp = config.get('mixed_precision', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Tensorboard
        log_dir = Path('runs') / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir)
        
        print(f"Trainer initialized:")
        print(f"  Model parameters: {count_parameters(self.model):,}")
        print(f"  Model size: {get_model_size_mb(self.model):.2f} MB")
        print(f"  Device: {device}")
        print(f"  Mixed precision: {self.use_amp}")
        print(f"  Loss function: {self.config['loss_type']}")
    
    def _setup_loss(self) -> nn.Module:
        """Setup loss function"""
        loss_type = self.config.get('loss_type', 'focal')
        
        if loss_type == 'focal':
            if self.config.get('label_smoothing', 0) > 0:
                return FocalLossWithLabelSmoothing(
                    alpha=self.config['focal_alpha'],
                    gamma=self.config['focal_gamma'],
                    epsilon=self.config['label_smoothing']
                )
            else:
                return FocalLoss(
                    alpha=self.config['focal_alpha'],
                    gamma=self.config['focal_gamma']
                )
        
        elif loss_type == 'cross_entropy':
            if self.config.get('label_smoothing', 0) > 0:
                return LabelSmoothingCrossEntropy(
                    epsilon=self.config['label_smoothing']
                )
            else:
                return nn.CrossEntropyLoss()
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer"""
        optimizer_type = self.config.get('optimizer', 'adamw').lower()
        
        if optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine').lower()
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
        else:
            return None
    
    def _progressive_unfreezing(self):
        """Implement progressive unfreezing strategy"""
        epoch = self.current_epoch
        
        if epoch < self.config.get('freeze_backbone_epochs', 10):
            # Freeze all backbone, train only head
            for name, param in self.model.named_parameters():
                if 'fc' not in name and 'classifier' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print(f"Epoch {epoch}: Training head only")
        
        elif epoch < self.config.get('unfreeze_late_blocks_epoch', 30):
            # Unfreeze late blocks
            for name, param in self.model.named_parameters():
                if any(x in name for x in ['features.6', 'features.7', 'features.8', 'fc', 'classifier']):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print(f"Epoch {epoch}: Training late blocks + head")
        
        else:
            # Unfreeze all
            for param in self.model.parameters():
                param.requires_grad = True
            if epoch == self.config.get('unfreeze_late_blocks_epoch', 30):
                print(f"Epoch {epoch}: Training full model")
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                
                if self.config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }
        
        # Save latest
        checkpoint_path = self.save_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = self.save_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model (val_acc: {self.best_val_acc:.2f}%)")
    
    def train(self, num_epochs: int = None):
        """Full training loop"""
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        print(f"\n{'='*60}")
        print(f"Starting Training for {num_epochs} epochs")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Progressive unfreezing
            self._progressive_unfreezing()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Log to tensorboard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.get('early_stopping_patience', 15):
                print(f"\n⚠️ Early stopping triggered (patience: {self.patience_counter})")
                break
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"  Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")
        
        self.writer.close()


def main():
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(SPLITS_DIR)
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        model_type='efficientnet',
        num_classes=DATASET_CONFIG['num_classes'],
        pretrained=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Train
    trainer.train()
    
    print("\n✅ Training script completed!")


if __name__ == "__main__":
    main()
