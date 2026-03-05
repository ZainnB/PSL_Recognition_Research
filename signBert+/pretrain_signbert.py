"""
Pretraining Script for SignBERT+
Run this separately from your main training pipeline.
"""

import torch
import sys
import os
from torch.utils.data import DataLoader

# Add path to import
sys.path.insert(0, os.path.dirname(__file__))

from signbert_plus import SignBERTPlus, SignBERTConfig, PretrainDataset, pretrain_collate_fn, pretrain_signbert

def main():
    # Config
    config = SignBERTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = SignBERTPlus(config)
    model.to(device)

    # Data
    data_dir = "Pose_extraction_tranfomer_PSL_pipeline-main/processed_psl_research_zain"
    dataset = PretrainDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=pretrain_collate_fn)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Pretrain
    print("Starting SignBERT+ pretraining...")
    pretrain_signbert(model, dataloader, optimizer, device, num_epochs=10)

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/signbert_pretrained.pt")
    print("Pretrained model saved to checkpoints/signbert_pretrained.pt")

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">c:\Users\22K-4508\Downloads\Pose_extraction_tranfomer_PSL_pipeline-main\pretrain_signbert.py