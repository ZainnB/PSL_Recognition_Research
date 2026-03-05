"""
Comprehensive Test Suite for PSL Transformer Pipeline
Tests data loading, model, GPU functionality before training
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np
from pathlib import Path

# Add the nested module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Pose_extraction_tranfomer_PSL_pipeline'))

from transformer import PSLTransformer, PositionalEncoding
from dataset_loader import PSLDataset, collate_fn

# ==========================================
# TEST CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "Pose_extraction_tranfomer_PSL_pipeline/processed_psl_research_zain"
BATCH_SIZE = 16
VERBOSE = True

def print_test(test_name):
    """Print test header."""
    print(f"\n{'='*60}")
    print(f"TEST: {test_name}")
    print('='*60)

def print_pass(message="✓ PASSED"):
    """Print pass message."""
    print(f"✓ {message}")

def print_fail(message="✗ FAILED"):
    """Print fail message."""
    print(f"✗ {message}")
    raise AssertionError(message)

# ==========================================
# TEST 1: DEVICE & CUDA SETUP
# ==========================================
def test_device_setup():
    print_test("Device & CUDA Setup")
    
    print(f"Device: {DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        
        # Test tensor allocation on GPU
        test_tensor = torch.randn(1).to(DEVICE)
        assert test_tensor.is_cuda, "Failed to allocate tensor on GPU"
        print_pass("GPU tensor allocation successful")
    else:
        print("⚠ CUDA not available, will use CPU")
    
    print_pass("Device setup verified")


# ==========================================
# TEST 2: POSITIONAL ENCODING
# ==========================================
def test_positional_encoding():
    print_test("Positional Encoding (Sinusoidal)")
    
    d_model = 256
    max_len = 200
    batch_size = 4
    seq_len = 150
    
    # Initialize positional encoding
    pos_encoder = PositionalEncoding(d_model, max_len).to(DEVICE)
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, d_model).to(DEVICE)
    output = pos_encoder(x)
    
    # Check shape
    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    print_pass(f"Output shape correct: {output.shape}")
    
    # Check that positional encoding is added
    assert not torch.allclose(output, x), "Positional encoding not applied"
    print_pass("Positional encoding correctly applied")
    
    # Check that encoding is non-zero
    assert torch.any(pos_encoder.pe != 0), "Positional encoding is zero"
    print_pass("Positional encoding is non-zero")
    
    # Check different sequence lengths
    for test_len in [50, 100, 150, 200]:
        x_test = torch.randn(2, test_len, d_model).to(DEVICE)
        out_test = pos_encoder(x_test)
        assert out_test.shape == (2, test_len, d_model), f"Failed for seq_len={test_len}"
    print_pass("Positional encoding works for various sequence lengths")


# ==========================================
# TEST 3: MODEL ARCHITECTURE
# ==========================================
def test_model_architecture():
    print_test("Model Architecture Consistency")
    
    num_classes = 10
    model = PSLTransformer(
        input_dim=294,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        num_classes=num_classes,
        dropout=0.1,
        max_len=200
    ).to(DEVICE)
    
    # Check model structure
    print(f"Model: {model.__class__.__name__}")
    print(f"Input Dimension: 294 (42 joints × 7 features)")
    print(f"Linear Projection: 294 → 256")
    print(f"Transformer Layers: 6")
    print(f"Number of Classes: {num_classes}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Verify layer structure
    assert isinstance(model.input_projection, nn.Linear), "Missing input projection layer"
    print_pass("Input projection layer (294→256) found")
    
    assert hasattr(model, 'pos_encoder'), "Missing positional encoder"
    print_pass("Positional encoder found")
    
    assert hasattr(model, 'transformer_encoder'), "Missing transformer encoder"
    print_pass("Transformer encoder found")
    
    assert hasattr(model, 'classifier'), "Missing classifier layer"
    print_pass("Classifier layer found")
    
    # Check that model is on correct device (type check)
    param_device = next(model.parameters()).device.type
    expected_device = DEVICE.type
    assert param_device == expected_device, f"Model device mismatch: {param_device} vs {expected_device}"
    print_pass(f"Model on correct device: {DEVICE}")


# ==========================================
# TEST 4: MODEL FORWARD PASS
# ==========================================
def test_model_forward_pass():
    print_test("Model Forward Pass")
    
    model = PSLTransformer(
        input_dim=294,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        num_classes=10,
        dropout=0.0,  # Disable dropout for testing
        max_len=200
    ).to(DEVICE)
    
    model.eval()
    
    # Test with different batch sizes and sequence lengths
    test_configs = [
        (1, 50, "Single sample, short sequence"),
        (4, 100, "Small batch, medium sequence"),
        (16, 150, "Standard batch, long sequence"),
        (8, 200, "Small batch, max length"),
    ]
    
    with torch.no_grad():
        for batch_size, seq_len, description in test_configs:
            poses = torch.randn(batch_size, seq_len, 294).to(DEVICE)
            attention_mask = torch.ones(batch_size, seq_len).to(DEVICE)
            
            logits = model(poses, attention_mask)
            
            # Check output shape
            assert logits.shape == (batch_size, 10), f"Output shape mismatch: {logits.shape}"
            
            # Check output is valid (no NaN or Inf)
            assert not torch.isnan(logits).any(), "NaN in output"
            assert not torch.isinf(logits).any(), "Inf in output"
            
            print_pass(f"{description}: {poses.shape} → {logits.shape}")
    
    print_pass("All forward pass tests completed successfully")


# ==========================================
# TEST 5: ATTENTION MASKING
# ==========================================
def test_attention_masking():
    print_test("Attention Masking (Padding Handling)")
    
    model = PSLTransformer(
        input_dim=294,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        num_classes=10,
        dropout=0.0,
        max_len=200
    ).to(DEVICE)
    
    model.eval()
    
    batch_size, max_seq_len = 4, 100
    poses = torch.randn(batch_size, max_seq_len, 294).to(DEVICE)
    
    with torch.no_grad():
        # Test 1: Full attention (all 1s)
        attention_mask_full = torch.ones(batch_size, max_seq_len).to(DEVICE)
        output_full = model(poses, attention_mask_full)
        print_pass("Forward pass with full attention mask")
        
        # Test 2: Partial attention (variable padding)
        attention_mask_partial = torch.ones(batch_size, max_seq_len).to(DEVICE)
        attention_mask_partial[0, 50:] = 0  # Pad second half of first sample
        attention_mask_partial[1, 75:] = 0  # Pad last quarter of second sample
        output_partial = model(poses, attention_mask_partial)
        print_pass("Forward pass with variable padding")
        
        # Test 3: Zero attention (all 0s should still work)
        attention_mask_zero = torch.zeros(1, max_seq_len).to(DEVICE)
        attention_mask_zero[0, 0] = 1  # At least one position valid
        output_zero = model(poses[:1], attention_mask_zero)
        print_pass("Forward pass with minimal attention")
        
        # Outputs should be different
        assert not torch.allclose(output_full, output_partial), \
            "Different masking should produce different outputs"
        print_pass("Different masking patterns produce different outputs")


# ==========================================
# TEST 6: DATASET & DATALOADER
# ==========================================
def test_dataset_and_dataloader():
    print_test("Dataset & DataLoader")
    
    if not os.path.exists(DATA_DIR):
        print(f"⚠ Data directory not found: {DATA_DIR}")
        print("Skipping dataset test")
        return
    
    # Create label map
    labels = set()
    for file in os.listdir(DATA_DIR):
        if file.endswith(".npy"):
            label = file.rsplit("_", 1)[0]
            labels.add(label)
    
    label_map = {label: idx for idx, label in enumerate(sorted(labels))}
    print_pass(f"Found {len(label_map)} classes")
    
    # Load dataset
    dataset = PSLDataset(DATA_DIR, label_map)
    print_pass(f"Dataset loaded: {len(dataset)} samples")
    
    # Test single batch
    poses, label, length = dataset[0]
    print_pass(f"Sample shape: {poses.shape}, Label: {label}, Length: {length}")
    assert poses.shape[1] == 294, f"Feature dimension mismatch: {poses.shape[1]} vs 294"
    print_pass("Feature dimension correct (294)")
    
    # Test DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    
    batch_poses, batch_labels, batch_masks = next(iter(dataloader))
    print_pass(f"DataLoader batch shapes:")
    print(f"  Poses: {batch_poses.shape}")
    print(f"  Labels: {batch_labels.shape}")
    print(f"  Masks: {batch_masks.shape}")
    
    # Verify shapes
    assert batch_poses.shape[2] == 294, "Feature dimension incorrect in batch"
    assert batch_labels.shape[0] == batch_poses.shape[0], "Batch size mismatch"
    assert batch_masks.shape == batch_poses.shape[:2], "Mask shape incorrect"
    print_pass("All batch shapes correct")


# ==========================================
# TEST 7: TRAINING STEP
# ==========================================
def test_training_step():
    print_test("Training Step (Forward + Backward)")
    
    model = PSLTransformer(
        input_dim=294,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        num_classes=10,
        dropout=0.1,
        max_len=200
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    # Create synthetic batch
    batch_size, seq_len, num_classes = 8, 100, 10
    poses = torch.randn(batch_size, seq_len, 294).to(DEVICE)
    labels = torch.randint(0, num_classes, (batch_size,)).to(DEVICE)
    attention_mask = torch.ones(batch_size, seq_len).to(DEVICE)
    
    # Forward pass
    logits = model(poses, attention_mask)
    assert logits.shape == (batch_size, num_classes), "Output shape mismatch"
    print_pass("Forward pass successful")
    
    # Compute loss
    loss = criterion(logits, labels)
    assert not torch.isnan(loss), "NaN loss detected"
    assert loss.item() > 0, "Loss should be positive"
    print_pass(f"Loss computed: {loss.item():.6f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    print_pass("Backward pass successful")
    
    # Check gradients
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and torch.any(param.grad != 0):
            has_gradients = True
            break
    assert has_gradients, "No gradients computed"
    print_pass("Gradients computed successfully")
    
    # Optimizer step
    optimizer.step()
    print_pass("Optimizer step successful")


# ==========================================
# TEST 8: GRADIENT CLIPPING
# ==========================================
def test_gradient_clipping():
    print_test("Gradient Clipping")
    
    model = PSLTransformer(
        input_dim=294,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        num_classes=10,
        dropout=0.1,
        max_len=200
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)  # Higher LR for testing
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    grad_clip_norm = 1.0
    
    # Create batch that might have large gradients
    poses = torch.randn(4, 50, 294).to(DEVICE) * 10  # Amplified input
    labels = torch.randint(0, 10, (4,)).to(DEVICE)
    attention_mask = torch.ones(4, 50).to(DEVICE)
    
    logits = model(poses, attention_mask)
    loss = criterion(logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Get max gradient before clipping
    max_grad_before = 0
    for param in model.parameters():
        if param.grad is not None:
            max_grad_before = max(max_grad_before, param.grad.abs().max().item())
    
    # Apply gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    
    # Get max gradient after clipping
    max_grad_after = 0
    for param in model.parameters():
        if param.grad is not None:
            max_grad_after = max(max_grad_after, param.grad.abs().max().item())
    
    assert max_grad_after <= grad_clip_norm + 1e-5, "Gradient clipping failed"
    print_pass(f"Gradient clipping successful: {max_grad_before:.6f} → {max_grad_after:.6f}")


# ==========================================
# TEST 9: MODEL DROPOUT
# ==========================================
def test_batch_normalization():
    print_test("Dropout & Regularization")
    
    model = PSLTransformer(
        input_dim=294,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        num_classes=10,
        dropout=0.5,  # High dropout for testing
        max_len=200
    ).to(DEVICE)
    
    poses = torch.randn(4, 100, 294).to(DEVICE)
    attention_mask = torch.ones(4, 100).to(DEVICE)
    
    # Training mode (dropout active)
    model.train()
    with torch.no_grad():
        output1 = model(poses, attention_mask)
        output2 = model(poses, attention_mask)
    
    # With dropout, outputs should be different
    is_different = not torch.allclose(output1, output2)
    if is_different:
        print_pass("Dropout active in training mode")
    else:
        print("⚠ Dropout may not be working properly")
    
    # Evaluation mode (dropout inactive)
    model.eval()
    with torch.no_grad():
        output3 = model(poses, attention_mask)
        output4 = model(poses, attention_mask)
    
    # Without dropout, outputs should be identical
    assert torch.allclose(output3, output4), "Outputs should be identical in eval mode"
    print_pass("Dropout inactive in eval mode (deterministic)")


# ==========================================
# MAIN TEST RUNNER
# ==========================================
def run_all_tests():
    print("\n" + "="*60)
    print("PSL TRANSFORMER TEST SUITE")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"PyTorch Version: {torch.__version__}")
    print("="*60)
    
    tests = [
        test_device_setup,
        test_positional_encoding,
        test_model_architecture,
        test_model_forward_pass,
        test_attention_masking,
        test_dataset_and_dataloader,
        test_training_step,
        test_gradient_clipping,
        test_batch_normalization,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print_fail(str(e))
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Ready for training.")
    else:
        print(f"\n⚠ {failed} test(s) failed. Please review errors above.")
    
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
