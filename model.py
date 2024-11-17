import torch
import torch.nn as nn
import math


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, hidden_d):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden_d)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_d, d_model)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_len):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_sequence_len, d_model)
        position = torch.arange(0, max_sequence_len, dtype=torch.float64)
        position = position.unsqueeze(1)

        div = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(
            div * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        print(f"x.dtype {x.dtype}")
        print(f"pe.dtype {self.pe.dtype}")
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_d, dropout):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=num_heads,
                                          dropout=0.2,
                                          batch_first=True)

        self.ffn = PositionWiseFFN(d_model, hidden_d)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_o = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_o))
        out = self.ffn(x)
        x = self.norm2(x + self.dropout(out))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_d, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model,
                                               num_heads=num_heads,
                                               dropout=dropout,
                                               batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model,
                                                num_heads=num_heads,
                                                dropout=dropout,
                                                batch_first=True)
        self.ffn = PositionWiseFFN(d_model, hidden_d)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        attn_o = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_o))

        attn_o = self.cross_attn(x, enc_out, enc_out,
                                 src_mask)
        x = self.norm2(x + self.dropout(attn_o))

        ff_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ff_out))
        return x


class MyTransformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int,
                 num_heads: int,
                 hidden_d: int,
                 dropout: float,
                 max_sequence_len: int,
                 num_layers: int):
        """
            Args:
                src_vocab_size: int, input vocabulary size
                tgt_vocab_size: int, target size
                d_model: int, embedding dimension
                num_heads: int, number of heads in MHA
                hidden_d: int, dimension of the feed forward
                dropout: int, dropout rate
                max_sequence_len: int, maximum sequence length
                num_layers: int, number of encoder and decoder layers
        """
        super(MyTransformer, self).__init__()
        self.encoder_embedding = nn.Linear(src_vocab_size,
                                           d_model)
        self.decoder_embedding = nn.Linear(tgt_vocab_size,
                                           d_model)
        self.positional_enc = PositionEncoding(d_model, max_sequence_len)

        self.encoder = nn.ModuleList(
                [EncoderLayer(d_model, num_heads, hidden_d, dropout)
                 for _ in range(num_layers)])
        self.decoder = nn.ModuleList(
                [DecoderLayer(d_model, num_heads, hidden_d, dropout)
                 for _ in range(num_layers)])

        self.fc = nn.Linear(in_features=d_model, out_features=tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nospeak_mask = (1 -
                        torch.triu(torch.ones(1, seq_length, seq_length),
                                   diagonal=1)).bool()
        print(f"tgt_mask.shape {tgt_mask.shape}")  # (16, 1, 71, 1, 4)
        print(f"nospeak_mask.shape {nospeak_mask.shape}")  # (1, 71, 71)
        tgt_mask = tgt_mask & nospeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        print(f"Transformer src.dtype {src.dtype}")
        mid = self.encoder_embedding(src)
        print(f"Transformer self.encoder_embedding(src).dtype {mid.dtype}")
        src_enc = self.positional_enc(mid)
        tgt_enc = self.positional_enc(self.decoder_embedding(tgt))
        src_embedded = self.dropout(src_enc)
        tgt_embedded = self.dropout(tgt_enc)

        enc_output = src_embedded
        for enc_layer in self.encoder:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output

# This is the basic implementation of a Transformer from paper "Attention is
# all you need"
