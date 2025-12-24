import torch
import torch.nn as nn
from core.config import Config
from torch.nn import functional as F
from models.layers import PatchEmbeddings, EncoderBlock, DecoderBlock, MLP
from data.processor import VOCAB_SIZE


class MathVision(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.n_embd = config.model.n_embd
        self.patch_emb = PatchEmbeddings(config=config)
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.model.num_layers)]
        )
        self.encoder_ln = nn.LayerNorm(config.model.n_embd)

        self.decoder_embd = nn.Embedding(VOCAB_SIZE, config.model.n_embd)
        self.decoder_pos_embd = nn.Parameter(
            torch.zeros(1, config.model.max_seq_len, config.model.n_embd)
        )
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(config) for _ in range(config.model.num_layers)]
        )
        self.decoder_ln = nn.LayerNorm(config.model.n_embd)
        self.head = nn.Linear(config.model.n_embd, VOCAB_SIZE)

    def generate_causal_mask(self, T, device):
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(self, images, decoder_input):
        x = self.patch_emb(images)
        for block in self.encoder_blocks:
            x = block(x)
        memory = self.encoder_ln(x)

        B, T = decoder_input.shape
        x = self.decoder_embd(decoder_input)
        x = x + self.decoder_pos_embd[:, :T]

        causal_mask = self.generate_causal_mask(T, device=x.device)
        for block in self.decoder_blocks:
            x = block(x, memory, causal_mask)

        x = self.decoder_ln(x)
        logits = self.head(x)

        return logits
