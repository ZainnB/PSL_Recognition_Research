"""
SignBERT+ Implementation: Hand-Model-Aware Self-Supervised Pretraining for Sign Language
Based on the paper: "SignBERT+: Hand-model-aware Self-supervised Pre-training for Sign Language Understanding"

This is a separate implementation from your pose extraction pipeline.
It takes pre-extracted pose sequences (.npy files) as input for pretraining and fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import math

# ==========================================
# CONFIG
# ==========================================
class SignBERTConfig:
    INPUT_DIM = 294  # 42 joints * 7 features (matches your pipeline)
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 6
    DIM_FEEDFORWARD = 1024
    DROPOUT = 0.1
    MAX_LEN = 200
    MASK_PROB = 0.15  # Mask 15% of tokens

# ==========================================
# EMBEDDING LAYER
# ==========================================
class PoseEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear = nn.Linear(config.INPUT_DIM, config.D_MODEL)
        self.pos_embedding = nn.Embedding(config.MAX_LEN, config.D_MODEL)
        self.dropout = nn.Dropout(config.DROPOUT)

    def forward(self, pose_seq, mask=None):
        # pose_seq: (batch, seq_len, INPUT_DIM)
        batch_size, seq_len, _ = pose_seq.shape

        # Linear projection
        x = self.linear(pose_seq)  # (batch, seq_len, D_MODEL)

        # Positional encoding
        positions = torch.arange(seq_len, device=pose_seq.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        # Apply mask if provided (for masking during pretraining)
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        return self.dropout(x)

# ==========================================
# TRANSFORMER ENCODER
# ==========================================
class SignBERTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS)

    def forward(self, x, src_key_padding_mask=None):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

# ==========================================
# HAND-MODEL-AWARE DECODER (Simplified)
# ==========================================
class HandModelDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # Simplified: Just a linear decoder (full MANO implementation would be complex)
        # In practice, you'd integrate MANO hand model here
        self.decoder = nn.Linear(config.D_MODEL, config.INPUT_DIM)

    def forward(self, encoded_seq):
        # encoded_seq: (batch, seq_len, D_MODEL)
        return self.decoder(encoded_seq)  # (batch, seq_len, INPUT_DIM)

# ==========================================
# SIGNBERT+ MODEL
# ==========================================
class SignBERTPlus(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = PoseEmbedding(config)
        self.encoder = SignBERTEncoder(config)
        self.decoder = HandModelDecoder(config)

    def forward(self, pose_seq, mask=None, src_key_padding_mask=None):
        # Embedding
        embedded = self.embedding(pose_seq, mask)

        # Encoding
        encoded = self.encoder(embedded, src_key_padding_mask)

        # Decoding (for reconstruction)
        reconstructed = self.decoder(encoded)

        return reconstructed

    def encode(self, pose_seq, src_key_padding_mask=None):
        # For fine-tuning: just return encoder output
        embedded = self.embedding(pose_seq)
        return self.encoder(embedded, src_key_padding_mask)

# ==========================================
# MASKING UTILITIES
# ==========================================
def create_mask(pose_seq, mask_prob=0.15):
    """
    Create masking for pretraining.
    Returns mask tensor: 1 for visible, 0 for masked
    """
    batch_size, seq_len, _ = pose_seq.shape
    mask = torch.rand(batch_size, seq_len) > mask_prob
    return mask.float()

def apply_mask_to_sequence(pose_seq, mask):
    """
    Apply mask to pose sequence for input to model
    """
    return pose_seq * mask.unsqueeze(-1)

# ==========================================
# DATASET FOR PRETRAINING
# ==========================================
class PretrainDataset(Dataset):
    def __init__(self, data_dir):
        self.samples = []
        for file in os.listdir(data_dir):
            if file.endswith(".npy"):
                path = os.path.join(data_dir, file)
                self.samples.append(path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        pose = np.load(path)  # (T, 294)
        pose = torch.tensor(pose, dtype=torch.float32)
        return pose

def pretrain_collate_fn(batch):
    # Pad sequences to max length
    max_len = max([seq.shape[0] for seq in batch])
    padded = []
    masks = []

    for seq in batch:
        T = seq.shape[0]
        pad_len = max_len - T
        if pad_len > 0:
            padded_seq = F.pad(seq, (0, 0, 0, pad_len))
        else:
            padded_seq = seq
        padded.append(padded_seq)

        # Attention mask: 1 for real, 0 for padded
        attn_mask = torch.ones(max_len)
        if pad_len > 0:
            attn_mask[T:] = 0
        masks.append(attn_mask)

    return torch.stack(padded), torch.stack(masks)

# ==========================================
# PRETRAINING FUNCTION
# ==========================================
def pretrain_signbert(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for pose_batch, attn_masks in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            pose_batch = pose_batch.to(device)
            attn_masks = attn_masks.to(device)

            # Create and apply mask
            mask = create_mask(pose_batch).to(device)
            masked_input = apply_mask_to_sequence(pose_batch, mask)

            # Forward pass
            reconstructed = model(masked_input, mask, attn_masks == 0)  # src_key_padding_mask

            # Loss: only on masked positions
            loss = criterion(reconstructed * (1 - mask).unsqueeze(-1), pose_batch * (1 - mask).unsqueeze(-1))
            loss = loss / (1 - mask).sum()  # Normalize by number of masked elements

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# ==========================================
# FINE-TUNING FOR CLASSIFICATION
# ==========================================
class SignClassifier(nn.Module):
    def __init__(self, signbert_encoder, num_classes):
        super().__init__()
        self.encoder = signbert_encoder
        self.classifier = nn.Linear(256, num_classes)  # Assuming pooled output

    def forward(self, pose_seq, attn_mask=None):
        encoded = self.encoder(pose_seq, attn_mask)
        # Simple mean pooling (you can use masked mean pooling like in your original)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    config = SignBERTConfig()
    model = SignBERTPlus(config)

    # Example pretraining
    dataset = PretrainDataset("Pose_extraction_tranfomer_PSL_pipeline-main/processed_psl_research_zain")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=pretrain_collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Pretrain
    pretrain_signbert(model, dataloader, optimizer, device, num_epochs=5)

    # Save pretrained model
    torch.save(model.state_dict(), "checkpoints/signbert_pretrained.pt")

    # For fine-tuning, load encoder and add classifier
    # (Similar to your original training script)</content>
<parameter name="filePath">c:\Users\22K-4508\Downloads\Pose_extraction_tranfomer_PSL_pipeline-main\signbert_plus.py