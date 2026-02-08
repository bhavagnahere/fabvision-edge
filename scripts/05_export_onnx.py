"""
Export trained PyTorch model to ONNX format
Required for Phase 1 deliverables and edge deployment
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATASET_CONFIG, MODELS_DIR, RESULTS_DIR
from scripts.models import create_model


class ONNXExporter:
    """Export PyTorch model to ONNX format"""
    
    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple = (1, 1, 224, 224),
        device: torch.device = torch.device('cpu')
    ):
        self.model = model.to(device)
        self.model.eval()
        self.input_shape = input_shape
        self.device = device
    
    def export(
        self,
        output_path: str,
        opset_version: int = 12,
        dynamic_axes: bool = True
    ):
        """
        Export model to ONNX format
        
        Args:
            output_path: Path to save ONNX model
            opset_version: ONNX opset version (12 for good compatibility)
            dynamic_axes: Whether to use dynamic batch size
        """
        print(f"\nExporting model to ONNX format...")
        print(f"  Input shape: {self.input_shape}")
        print(f"  Opset version: {opset_version}")
        print(f"  Dynamic axes: {dynamic_axes}")
        
        # Create dummy input
        dummy_input = torch.randn(self.input_shape).to(self.device)
        
        # Define dynamic axes if enabled
        if dynamic_axes:
            dynamic_axes_dict = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        else:
            dynamic_axes_dict = None
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes_dict
        )
        
        print(f"✅ Model exported to {output_path}")
        
        # Get file size
        file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"   Model size: {file_size_mb:.2f} MB")
    
    def verify_onnx(self, onnx_path: str) -> bool:
        """Verify ONNX model is valid"""
        print(f"\nVerifying ONNX model...")
        
        try:
            # Load ONNX model
            onnx_model = onnx.load(onnx_path)
            
            # Check model
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model is valid")
            
            # Print model info
            print(f"\nModel Information:")
            print(f"  IR version: {onnx_model.ir_version}")
            print(f"  Producer: {onnx_model.producer_name}")
            print(f"  Graph name: {onnx_model.graph.name}")
            print(f"  Inputs: {len(onnx_model.graph.input)}")
            print(f"  Outputs: {len(onnx_model.graph.output)}")
            print(f"  Nodes: {len(onnx_model.graph.node)}")
            
            return True
        
        except Exception as e:
            print(f"❌ ONNX model verification failed: {e}")
            return False
    
    def test_onnx_inference(self, onnx_path: str) -> bool:
        """Test ONNX model inference"""
        print(f"\nTesting ONNX inference...")
        
        try:
            # Create ONNX Runtime session
            ort_session = ort.InferenceSession(onnx_path)
            
            # Get input/output names
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            
            print(f"  Input name: {input_name}")
            print(f"  Output name: {output_name}")
            
            # Create dummy input
            dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
            
            # Run inference
            ort_outputs = ort_session.run([output_name], {input_name: dummy_input})
            
            print(f"  Output shape: {ort_outputs[0].shape}")
            print(f"  Output range: [{ort_outputs[0].min():.3f}, {ort_outputs[0].max():.3f}]")
            
            # Compare with PyTorch output
            with torch.no_grad():
                pytorch_input = torch.from_numpy(dummy_input).to(self.device)
                pytorch_output = self.model(pytorch_input).cpu().numpy()
            
            # Check difference
            max_diff = np.abs(pytorch_output - ort_outputs[0]).max()
            print(f"  Max difference (PyTorch vs ONNX): {max_diff:.6f}")
            
            if max_diff < 1e-4:
                print("✅ ONNX inference matches PyTorch (difference < 1e-4)")
                return True
            elif max_diff < 1e-2:
                print("⚠️  ONNX inference has small differences (< 1e-2)")
                return True
            else:
                print("❌ ONNX inference differs significantly from PyTorch")
                return False
        
        except Exception as e:
            print(f"❌ ONNX inference test failed: {e}")
            return False
    
    def benchmark_onnx(self, onnx_path: str, num_runs: int = 100):
        """Benchmark ONNX inference speed"""
        print(f"\nBenchmarking ONNX inference ({num_runs} runs)...")
        
        try:
            import time
            
            # Create session
            ort_session = ort.InferenceSession(onnx_path)
            input_name = ort_session.get_inputs()[0].name
            output_name = ort_session.get_outputs()[0].name
            
            # Dummy input
            dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
            
            # Warmup
            for _ in range(10):
                ort_session.run([output_name], {input_name: dummy_input})
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.time()
                ort_session.run([output_name], {input_name: dummy_input})
                times.append((time.time() - start) * 1000)  # Convert to ms
            
            times = np.array(times)
            
            print(f"\nInference Time Statistics:")
            print(f"  Mean: {times.mean():.2f} ms")
            print(f"  Median: {np.median(times):.2f} ms")
            print(f"  Std: {times.std():.2f} ms")
            print(f"  Min: {times.min():.2f} ms")
            print(f"  Max: {times.max():.2f} ms")
            print(f"  P95: {np.percentile(times, 95):.2f} ms")
            print(f"  P99: {np.percentile(times, 99):.2f} ms")
            print(f"  Throughput: {1000 / times.mean():.1f} images/sec")
        
        except Exception as e:
            print(f"❌ Benchmarking failed: {e}")


def main():
    """Main export function"""
    print("="*60)
    print("FabVision Edge - ONNX Export")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print("\nLoading trained model...")
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
    print(f"   Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    # Create exporter
    input_shape = (1, 1, 224, 224)  # (batch, channels, height, width)
    exporter = ONNXExporter(
        model=model,
        input_shape=input_shape,
        device=device
    )
    
    # Export to ONNX
    output_dir = Path(MODELS_DIR) / 'onnx'
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / 'defect_classifier.onnx'
    
    exporter.export(
        output_path=str(onnx_path),
        opset_version=12,
        dynamic_axes=True
    )
    
    # Verify ONNX model
    is_valid = exporter.verify_onnx(str(onnx_path))
    
    if not is_valid:
        print("\n❌ ONNX export failed verification")
        return
    
    # Test inference
    inference_ok = exporter.test_onnx_inference(str(onnx_path))
    
    if not inference_ok:
        print("\n⚠️  ONNX inference test showed differences")
    
    # Benchmark
    exporter.benchmark_onnx(str(onnx_path), num_runs=100)
    
    # Save metadata
    metadata = {
        'model_type': 'efficientnet_lite3_with_attention',
        'num_classes': DATASET_CONFIG['num_classes'],
        'class_names': DATASET_CONFIG['class_names'],
        'input_shape': list(input_shape),
        'opset_version': 12,
        'best_val_accuracy': float(checkpoint['best_val_acc']),
        'trained_epoch': int(checkpoint['epoch']),
        'model_size_mb': float(Path(onnx_path).stat().st_size / (1024 * 1024))
    }
    
    metadata_path = output_dir / 'model_metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Metadata saved to {metadata_path}")
    
    print("\n" + "="*60)
    print("ONNX Export Complete!")
    print(f"  Model: {onnx_path}")
    print(f"  Size: {metadata['model_size_mb']:.2f} MB")
    print("="*60)


if __name__ == "__main__":
    main()
