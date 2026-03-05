"""
Quick Diagnostic Tests - GPU, Dependencies, and Environment Check
"""

import torch
import sys
import os

print("\n" + "="*60)
print("DIAGNOSTIC TEST SUITE")
print("="*60)

# Test 1: PyTorch & CUDA
print("\n[1] PyTorch Installation")
print(f"  PyTorch Version: {torch.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    
    # Test GPU memory
    try:
        x = torch.randn(1000, 1000).cuda()
        del x
        print("  GPU Memory: OK")
    except Exception as e:
        print(f"  GPU Memory: ERROR - {e}")
else:
    print("  WARNING: CUDA not available, CPU will be used")

# Test 2: Required Libraries
print("\n[2] Required Libraries")
required_libs = {
    'torch': torch,
    'torch.nn': 'torch.nn',
    'tensorboard': None,
    'tqdm': None,
    'numpy': None,
    'sklearn': None,
}

for lib_name in required_libs.keys():
    try:
        if lib_name.startswith('torch'):
            print(f"  ✓ {lib_name}")
        else:
            __import__(lib_name)
            print(f"  ✓ {lib_name}")
    except ImportError:
        print(f"  ✗ {lib_name} NOT FOUND")

# Test 3: Custom Modules
print("\n[3] Custom Modules")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Pose_extraction_tranfomer_PSL_pipeline-main'))

try:
    from transformer import PSLTransformer, PositionalEncoding
    print("  ✓ transformer.py")
except ImportError as e:
    print(f"  ✗ transformer.py - {e}")

try:
    from dataset_loader import PSLDataset, collate_fn
    print("  ✓ dataset_loader.py")
except ImportError as e:
    print(f"  ✗ dataset_loader.py - {e}")

# Test 4: Data Directory
print("\n[4] Data Directory")
DATA_DIR = "Pose_extraction_tranfomer_PSL_pipeline-main/processed_psl_research_zain"
if os.path.exists(DATA_DIR):
    npy_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')]
    print(f"  ✓ Data directory found")
    print(f"  Total samples: {len(npy_files)}")
else:
    print(f"  ✗ Data directory not found: {DATA_DIR}")

print("\n" + "="*60)
print("Run 'python test_pipeline.py' for comprehensive tests")
print("="*60 + "\n")