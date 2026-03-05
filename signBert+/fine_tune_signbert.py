"""
Fine-tuning Script for SignBERT+ on PSL Classification
Uses the pretrained encoder from pretrain_signbert.py
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
sys.path.insert(0, os.path.dirname(__file__))

from dataset_loader import PSLDataset, collate_fn
from signbert_plus import SignBERTPlus, SignBERTConfig, SignClassifier

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # Paths
    DATA_DIR = "Pose_extraction_tranfomer_PSL_pipeline-main/processed_psl_research_zain"
    CHECKPOINT_DIR = "checkpoints"
    LOGS_DIR = "logs"
    PRETRAINED_PATH = "checkpoints/signbert_pretrained.pt"

    # Model Architecture
    NUM_CLASSES = 100  # Adjust based on your labels
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0

def create_label_map(data_dir):
    """Create label map from your data directory"""
    labels = set()
    for file in os.listdir(data_dir):
        if file.endswith(".npy"):
            label = file.rsplit("_", 1)[0]
            labels.add(label)
    return {label: i for i, label in enumerate(sorted(labels))}

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(dataloader, desc="Training"):
        poses, labels, attention_masks = batch
        poses = poses.to(device)
        labels = labels.to(device)
        attention_masks = attention_masks.to(device)

        optimizer.zero_grad()
        outputs = model(poses, attention_masks == 0)  # src_key_padding_mask
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), correct / total

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            poses, labels, attention_masks = batch
            poses = poses.to(device)
            labels = labels.to(device)
            attention_masks = attention_masks.to(device)

            outputs = model(poses, attention_masks == 0)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), correct / total

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create label map
    label_map = create_label_map(config.DATA_DIR)
    config.NUM_CLASSES = len(label_map)

    # Dataset
    dataset = PSLDataset(config.DATA_DIR, label_map)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Load pretrained SignBERT+
    signbert_config = SignBERTConfig()
    pretrained_model = SignBERTPlus(signbert_config)
    pretrained_model.load_state_dict(torch.load(config.PRETRAINED_PATH))
    pretrained_model.to(device)

    # Create classifier with pretrained encoder
    model = SignClassifier(pretrained_model.encoder, config.NUM_CLASSES)
    model.to(device)

    # Only train the classifier head (freeze encoder)
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Optimizer (only for classifier)
    optimizer = optim.Adam(model.classifier.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(".4f")
        print(".4f")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(), f"{config.CHECKPOINT_DIR}/signbert_finetuned_best.pt")

    # Save history
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{config.LOGS_DIR}/signbert_finetune_history_{timestamp}.json", "w") as f:
        json.dump(history, f)

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\22K-4508\Downloads\Pose_extraction_tranfomer_PSL_pipeline-main\fine_tune_signbert.py