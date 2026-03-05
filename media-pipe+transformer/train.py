"""
Pose Sign Language (PSL) Transformer Training Pipeline
Architecture: Linear(294→256) + Sinusoidal PE + 6 TransformerLayers + Masked_Mean_Pooling + Linear
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime

# Add the nested module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Pose_extraction_tranfomer_PSL_pipeline-main'))

from transformer import PSLTransformer
from dataset_loader import PSLDataset, collate_fn

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Paths
    DATA_DIR = "Pose_extraction_tranfomer_PSL_pipeline/processed_psl_research_zain"
    CHECKPOINT_DIR = "checkpoints"
    LOGS_DIR = "logs"
    
    # Model Architecture (FIXED - matches your spec)
    INPUT_DIM = 294  # 42 joints * 7 features (x,y,z, vx,vy,vz, conf)
    D_MODEL = 256    # Linear projection output
    NHEAD = 8
    NUM_LAYERS = 6   # Exactly 6 transformer layers
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    MAX_LEN = 200
    
    # Training
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    
    # Data
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    RANDOM_SEED = 42
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Logging
    LOG_INTERVAL = 10
    SAVE_INTERVAL = 5


# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def create_label_map(data_dir):
    """Extract unique labels from dataset and create label mapping."""
    labels = set()
    
    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            # Extract label from filename (everything before the last underscore/number)
            label = file.rsplit("_", 1)[0]
            labels.add(label)
    
    label_map = {label: idx for idx, label in enumerate(sorted(labels))}
    reverse_label_map = {idx: label for label, idx in label_map.items()}
    
    return label_map, reverse_label_map


def init_directories():
    """Create necessary directories."""
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.LOGS_DIR, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, metrics, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': {
            'input_dim': Config.INPUT_DIM,
            'd_model': Config.D_MODEL,
            'nhead': Config.NHEAD,
            'num_layers': Config.NUM_LAYERS,
            'dim_feedforward': Config.DIM_FEEDFORWARD,
            'dropout': Config.DROPOUT,
        }
    }
    torch.save(checkpoint, save_path)
    print(f"✓ Checkpoint saved: {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"✓ Checkpoint loaded from {checkpoint_path}")
    return model, optimizer, start_epoch, checkpoint.get('metrics', {})


def print_model_summary(model, num_classes):
    """Print model architecture summary."""
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*60)
    print(f"Input Dimension:           {Config.INPUT_DIM}")
    print(f"Linear Projection:         {Config.INPUT_DIM} → {Config.D_MODEL}")
    print(f"Positional Encoding:       Sinusoidal (max_len={Config.MAX_LEN})")
    print(f"Transformer Layers:        {Config.NUM_LAYERS} × (d_model={Config.D_MODEL}, nhead={Config.NHEAD})")
    print(f"Feedforward Dimension:     {Config.DIM_FEEDFORWARD}")
    print(f"Dropout:                   {Config.DROPOUT}")
    print(f"Pooling:                   Masked Mean Pooling")
    print(f"Output Projection:         {Config.D_MODEL} → {num_classes}")
    print(f"Total Parameters:          {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters:      {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("="*60 + "\n")


# ==========================================
# TRAINING & VALIDATION
# ==========================================
def train_epoch(model, train_loader, optimizer, criterion, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for poses, labels, attention_mask in pbar:
        poses = poses.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        attention_mask = attention_mask.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        logits = model(poses, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy calculation
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def validate_epoch(model, val_loader, criterion, config):
    """Validate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    for poses, labels, attention_mask in pbar:
        poses = poses.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        attention_mask = attention_mask.to(config.DEVICE)
        
        logits = model(poses, attention_mask)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


@torch.no_grad()
def evaluate_test(model, test_loader, criterion, config, reverse_label_map):
    """Evaluate on test set with detailed metrics."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    class_correct = {cls: 0 for cls in reverse_label_map.values()}
    class_total = {cls: 0 for cls in reverse_label_map.values()}
    
    pbar = tqdm(test_loader, desc="Testing", leave=False)
    for poses, labels, attention_mask in pbar:
        poses = poses.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        attention_mask = attention_mask.to(config.DEVICE)
        
        logits = model(poses, attention_mask)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Per-class accuracy
        for i, label in enumerate(labels):
            class_name = reverse_label_map[label.item()]
            class_total[class_name] += 1
            if predicted[i] == label:
                class_correct[class_name] += 1
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    print("\nPer-Class Test Accuracy:")
    for class_name in sorted(class_total.keys()):
        if class_total[class_name] > 0:
            acc = 100 * class_correct[class_name] / class_total[class_name]
            print(f"  {class_name:30s}: {acc:6.2f}% ({class_correct[class_name]}/{class_total[class_name]})")
    
    return avg_loss, accuracy


# ==========================================
# MAIN TRAINING FUNCTION
# ==========================================
def main(args):
    print(f"\n{'='*60}")
    print(f"PSL Transformer Training Pipeline")
    print(f"Device: {Config.DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"{'='*60}\n")
    
    # Initialize
    init_directories()
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    # Check data directory
    if not os.path.exists(Config.DATA_DIR):
        print(f"ERROR: Data directory not found: {Config.DATA_DIR}")
        print("Please ensure the processed dataset is available.")
        sys.exit(1)
    
    # Load dataset
    print("Loading dataset...")
    label_map, reverse_label_map = create_label_map(Config.DATA_DIR)
    num_classes = len(label_map)
    
    print(f"Found {num_classes} classes: {list(label_map.keys())}")
    
    dataset = PSLDataset(Config.DATA_DIR, label_map)
    print(f"Total samples: {len(dataset)}")
    
    # Train/Val/Test split
    train_size = int(Config.TRAIN_SPLIT * len(dataset))
    val_size = int(Config.VAL_SPLIT * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(Config.RANDOM_SEED)
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = PSLTransformer(
        input_dim=Config.INPUT_DIM,
        d_model=Config.D_MODEL,
        nhead=Config.NHEAD,
        num_layers=Config.NUM_LAYERS,
        dim_feedforward=Config.DIM_FEEDFORWARD,
        num_classes=num_classes,
        dropout=Config.DROPOUT,
        max_len=Config.MAX_LEN
    ).to(Config.DEVICE)
    
    print_model_summary(model, num_classes)
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # Load checkpoint if provided
    start_epoch = 0
    best_val_acc = 0
    if args.resume:
        if os.path.exists(args.resume):
            model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, args.resume)
        else:
            print(f"WARNING: Checkpoint not found: {args.resume}")
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    print(f"\n{'='*60}")
    print(f"Starting Training: {Config.NUM_EPOCHS} epochs")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Learning Rate: {Config.LEARNING_RATE}")
    print(f"{'='*60}\n")
    
    # Training loop
    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, Config)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, Config)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % Config.SAVE_INTERVAL == 0:
            checkpoint_path = os.path.join(
                Config.CHECKPOINT_DIR,
                f"model_epoch_{epoch+1:03d}_val_acc_{val_acc:.2f}.pt"
            )
            save_checkpoint(model, optimizer, epoch, history, checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "model_best.pt")
            save_checkpoint(model, optimizer, epoch, history, best_checkpoint_path)
            print(f"  ★ New best model saved (Val Acc: {val_acc:.2f}%)")
    
    # Test
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")
    
    # Load best model for testing
    if os.path.exists(os.path.join(Config.CHECKPOINT_DIR, "model_best.pt")):
        model, _, _, _ = load_checkpoint(
            model, optimizer,
            os.path.join(Config.CHECKPOINT_DIR, "model_best.pt")
        )
    
    test_loss, test_acc = evaluate_test(model, test_loader, criterion, Config, reverse_label_map)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Save history
    history_file = os.path.join(Config.LOGS_DIR, f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✓ Training history saved to {history_file}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PSL Transformer")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    main(args)