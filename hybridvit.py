import torch
import torch.nn as nn


# 5 classes to create


# Patchify
class Embedding(nn.Module):
    def __init__(self, in_features, d_model):
        super(Embedding, self).__init__()

        self.projection = nn.Sequential(
            nn.LayerNorm(100),
            nn.Conv1d(in_features, 16, 5),
            nn.ReLU(),
            nn.LayerNorm(50),
            nn.Conv1d(16, d_model, 5),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape -> (B, Seq_len, in_features)
        x = x.permute(0, 2, 1).contiguous()
        # x shape -> (B, in_features, Seq_len)
        x = self.projection(x)
        # x shape -> (B, seq_len, d_m)
        return x


# PositionalEncoding (w/ cls token)
class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(PositionalEncoding, self).__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model),
                                      requires_grad=True)
        self.pe = nn.Parameter(torch.randn(1, seq_len + 1, d_model),
                               requires_grad=True)

    def forward(self, x):
        # x shape -> (B, seq_len, d_m)
        # x = x.squeeze(1)

        B, seq_len, _ = x.shape
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        # x shape -> (B, seq_len + 1, d_m)
        x = x + self.pe[:, : seq_len + 1]
        return x


# AttentionHead ( I am not going to be implementing this from scratch)
# MultiHeadAttention (Because of performance issues)
# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ff_d, dropout):
        super(EncoderLayer, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(ff_d),
            nn.Linear(ff_d, d_model),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape -> (B, seq_len + 1, d_model)
        att_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(att_out))
        ff_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


# TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, ff_d, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        self.encoders = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, ff_d, dropout)
             for _ in range(num_layers)]
        )

    def forward(self, x):
        # x shape -> (B, seq_len + 1, d_model)
        enc_out = x
        for enc in self.encoders:
            enc_out = enc(enc_out)
        return enc_out


# ViT
class ViT(nn.Module):
    def __init__(
        self,
        in_features,
        d_model,
        seq_len,
        num_heads,
        ff_d,
        num_layers,
        num_classes,
        dropout,
    ):
        super(ViT, self).__init__()

        self.embedding = Embedding(in_features, d_model)
        self.pe = PositionalEncoding(seq_len, d_model)

        self.transformer = TransformerEncoder(
            d_model, num_heads, ff_d, num_layers, dropout
        )
        self.head = nn.Sequential(
            nn.Linear(d_model, ff_d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(ff_d),
            nn.Linear(ff_d, num_classes),
        )

    def forward(self, x):
        embed = self.embedding(x)
        out = self.pe(embed)
        out = self.transformer(out)
        # out shape -> (B, seq_len + 1, d_model)
        out = self.head(out[:, 0, :]).squeeze()
        # out shape -> (B, num_classes)
        return out
