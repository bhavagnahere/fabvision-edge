"""
Inference script for testing single images
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATASET_CONFIG, MODELS_DIR
from scripts.models import create_model


class DefectPredictor:
    """Single image defect prediction"""
    
    def __init__(self, model_path: str, device: torch.device = None):
        self.class_names = DATASET_CONFIG['class_names']
        self.image_size = DATASET_CONFIG['image_size']
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Load model
        self.model = create_model(
            model_type='efficientnet',
            num_classes=len(self.class_names),
            pretrained=False
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Get normalization stats from checkpoint if available
        self.mean = 0.456
        self.std = 0.224
        
        print(f"✅ Model loaded from {model_path}")
        print(f"   Device: {self.device}")
        print(f"   Classes: {self.class_names}")
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for inference"""
        # Load as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
        
        return img_tensor
    
    def predict(self, image_path: str, top_k: int = 3):
        """
        Predict defect class for an image
        
        Args:
            image_path: Path to image file
            top_k: Return top-k predictions
        
        Returns:
            Dictionary with predictions
        """
        # Preprocess
        img_tensor = self.preprocess_image(image_path)
        img_tensor = img_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)[0]
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            predictions.append({
                'class': self.class_names[idx.item()],
                'probability': prob.item(),
                'confidence_pct': prob.item() * 100
            })
        
        return {
            'top_prediction': predictions[0],
            'top_k_predictions': predictions,
            'all_probabilities': {
                name: float(probabilities[i])
                for i, name in enumerate(self.class_names)
            }
        }
    
    def predict_batch(self, image_paths: list):
        """Predict for multiple images"""
        results = []
        
        for img_path in image_paths:
            print(f"\nPredicting: {img_path}")
            try:
                result = self.predict(img_path)
                results.append({
                    'image': img_path,
                    'success': True,
                    **result
                })
                
                # Print result
                top = result['top_prediction']
                print(f"  → {top['class']} ({top['confidence_pct']:.2f}%)")
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    'image': img_path,
                    'success': False,
                    'error': str(e)
                })
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Defect Classification Inference')
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image or directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model checkpoint (default: models/saved/checkpoint_best.pth)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Show top-k predictions (default: 3)'
    )
    
    args = parser.parse_args()
    
    # Set model path
    if args.model is None:
        model_path = Path(MODELS_DIR) / 'saved' / 'checkpoint_best.pth'
    else:
        model_path = Path(args.model)
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first or specify correct path with --model")
        return
    
    # Create predictor
    predictor = DefectPredictor(str(model_path))
    
    # Check if input is directory or single file
    input_path = Path(args.image)
    
    if input_path.is_dir():
        # Process all images in directory
        image_files = list(input_path.glob('*.png')) + \
                     list(input_path.glob('*.jpg')) + \
                     list(input_path.glob('*.jpeg'))
        
        print(f"\nFound {len(image_files)} images in {input_path}")
        results = predictor.predict_batch([str(f) for f in image_files])
        
    elif input_path.is_file():
        # Process single image
        result = predictor.predict(str(input_path), top_k=args.top_k)
        
        print("\n" + "="*60)
        print(f"Image: {input_path.name}")
        print("="*60)
        
        print(f"\n🎯 Top Prediction:")
        top = result['top_prediction']
        print(f"   Class: {top['class']}")
        print(f"   Confidence: {top['confidence_pct']:.2f}%")
        
        if args.top_k > 1:
            print(f"\n📊 Top-{args.top_k} Predictions:")
            for i, pred in enumerate(result['top_k_predictions'], 1):
                print(f"   {i}. {pred['class']}: {pred['confidence_pct']:.2f}%")
        
        print("\n📈 All Class Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            bar = '█' * int(prob * 50)
            print(f"   {class_name:15s} {bar} {prob*100:5.2f}%")
        
        print("\n" + "="*60)
    
    else:
        print(f"❌ Invalid path: {input_path}")


if __name__ == "__main__":
    main()
