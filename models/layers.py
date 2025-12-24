import torch
import torch.nn as nn
from torch.nn import functional as F
from core.config import Config


class PatchEmbeddings(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.patch_size = config.model.patch_size
        self.img_size = config.data.image_size
        self.n_embd = config.model.n_embd
        self.grid_size = self.img_size // self.patch_size
        self.proj = nn.Conv2d(
            in_channels=1,
            out_channels=self.n_embd,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.row_embd = nn.Embedding(self.grid_size, self.n_embd)
        self.col_embd = nn.Embedding(self.grid_size, self.n_embd)
        self.ln = nn.LayerNorm(self.n_embd)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)
        gh, gw = self.grid_size, self.grid_size

        rows = self.row_embd(torch.arange(gh, device=x.device)).view(gh, 1, self.n_embd)
        cols = self.col_embd(torch.arange(gw, device=x.device)).view(1, gw, self.n_embd)

        pos_emb_2d = (rows + cols).view(-1, self.n_embd)

        return self.ln(x + pos_emb_2d)


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        n_embd = config.model.n_embd
        hidden_dim = config.model.dim_ratio * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.model.dropout),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(config.model.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_embd = config.model.n_embd
        self.ln1 = nn.LayerNorm(self.n_embd)
        self.ln2 = nn.LayerNorm(self.n_embd)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.n_embd,
            num_heads=config.model.num_heads,
            dropout=config.model.dropout,
            batch_first=True,
        )
        self.mlp = MLP(config=config)
        self.dropout = nn.Dropout(config.model.dropout)

    def forward(self, x, return_attn=False, context=None, mask=None):
        x_norm = self.ln1(x)

        kv = x_norm if context is None else context
        attn_out, weights = self.attn(
            x_norm, kv, kv, need_weights=return_attn, attn_mask=mask
        )
        x = x + self.dropout(attn_out)
        x = x + self.mlp(self.ln2(x))
        if return_attn:
            return x, weights
        return x
