import torch
from torch.utils.data import Dataset
import os
import numpy as np

class PSLDataset(Dataset):
    def __init__(self, data_dir, label_map):
        self.samples = []
        self.label_map = label_map

        for file in os.listdir(data_dir):
            if file.endswith(".npy"):
                label = file.rsplit("_", 1)[0]
                path = os.path.join(data_dir, file)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        pose = np.load(path)  # (T, 42, 7)
        T = pose.shape[0]

        pose = pose.reshape(T, -1)  # (T, 294)

        pose = torch.tensor(pose, dtype=torch.float32)
        label = torch.tensor(self.label_map[label], dtype=torch.long)

        return pose, label, T
    

def collate_fn(batch):
    poses, labels, lengths = zip(*batch)

    max_len = max(lengths)
    feature_dim = poses[0].shape[1]

    padded = torch.zeros(len(batch), max_len, feature_dim)
    attention_mask = torch.zeros(len(batch), max_len)

    for i, pose in enumerate(poses):
        T = pose.shape[0]
        padded[i, :T] = pose
        attention_mask[i, :T] = 1

    labels = torch.stack(labels)

    return padded, labels, attention_mask