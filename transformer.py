import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, d_model)
        T = x.size(1)
        return x + self.pe[:, :T]


class PSLTransformer(nn.Module):
    def __init__(
        self,
        input_dim=294,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        num_classes=10,
        dropout=0.1,
        max_len=200
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, attention_mask):
        """
        x: (B, T, 294)
        attention_mask: (B, T) with 1 for valid, 0 for pad
        """

        # Project input
        x = self.input_projection(x)  # (B, T, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create key padding mask (True = ignore position)
        key_padding_mask = attention_mask == 0  # (B, T)

        # Transformer
        x = self.transformer_encoder(
            x,
            src_key_padding_mask=key_padding_mask
        )  # (B, T, d_model)

        # Masked Mean Pooling
        attention_mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
        x = x * attention_mask

        sum_embeddings = x.sum(dim=1)  # (B, d_model)
        lengths = attention_mask.sum(dim=1)  # (B, 1)

        mean_pooled = sum_embeddings / lengths.clamp(min=1e-6)

        out = self.dropout(mean_pooled)
        logits = self.classifier(out)

        return logits