"""
gpt_model.py  —  Introduction to AI 
Task 2: Character-Level Generative Transformer (GPT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models import AttentionBlock


class Transformer_Block(nn.Module):
    """
    A single Transformer block applying six steps:
        1. Self-attention
        2. Residual add
        3. Layer norm
        4. Feed-forward + ReLU
        5. Residual add
        6. Layer norm
    """

    def __init__(self, layer_size: int):
        super().__init__()
        self.attention = AttentionBlock(layer_size)
        self.norm1     = nn.LayerNorm(layer_size)
        self.norm2     = nn.LayerNorm(layer_size)
        self.ff        = nn.Linear(layer_size, layer_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor (batch_size, seq_len, layer_size)
        Returns:
            FloatTensor (batch_size, seq_len, layer_size)
        """

        # Step 1 — Self-attention
        attn_out = self.attention(x)

        # Step 2 — Residual add (add to the ORIGINAL input x)
        x2 = attn_out + x

        # Step 3 — Layer norm
        x3 = self.norm1(x2)

        # Step 4 — Feed-forward with ReLU
        ff_out = torch.relu(self.ff(x3))

        # Step 5 — Residual add (add to output of step 3, not step 4)
        x5 = ff_out + x3

        # Step 6 — Layer norm
        out = self.norm2(x5)

        return out



class GPT(nn.Module):
    """
    Character-level generative Transformer.

    Architecture:
        1. Embedding
        2. N x Transformer blocks
        3. LayerNorm
        4. Linear projection (NO activation)
    """

    def __init__(
        self,
        vocab_size:  int,
        block_size:  int,
        layer_size:  int = 64,
        n_layers:    int = 2,
    ):
        super().__init__()
        self.block_size  = block_size
        self.vocab_size  = vocab_size

        self.embedding   = nn.Embedding(vocab_size, layer_size)
        self.blocks      = nn.ModuleList(
            [Transformer_Block(layer_size) for _ in range(n_layers)]
        )
        self.norm        = nn.LayerNorm(layer_size)
        self.output_proj = nn.Linear(layer_size, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: LongTensor (batch_size, seq_len)
        Returns:
            logits: FloatTensor (batch_size, seq_len, vocab_size)
        """

        # Step 1 — Embed token indices into dense vectors
        # idx: (batch, seq_len) -> x: (batch, seq_len, layer_size)
        x = self.embedding(idx)

        # Step 2 — Pass through each Transformer block sequentially
        for block in self.blocks:
            x = block(x)

        # Step 3 — Final layer norm
        x = self.norm(x)

        # Step 4 — Project to vocabulary logits
        # NO activation here — raw logits only
        logits = self.output_proj(x)   # (batch, seq_len, vocab_size)

        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Autoregressively generate max_new_tokens new characters.
        Already implemented — do not modify.
        """
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -self.block_size:]
            logits    = self(idx_cond)
            logits    = logits[:, -1, :]
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, idx_next], dim=1)
        return idx



if __name__ == '__main__':
    print("Running shape tests...")

    vocab_size = 50
    block_size = 16
    layer_size = 32
    n_layers   = 2
    batch_size = 4

    print("\n  Testing Transformer_Block...")
    tb  = Transformer_Block(layer_size)
    x   = torch.randn(batch_size, block_size, layer_size)
    out = tb(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"  Transformer_Block: {x.shape} -> {out.shape}  \u2713")

    print("\n  Testing GPT...")
    model  = GPT(vocab_size, block_size, layer_size, n_layers)
    idx    = torch.randint(0, vocab_size, (batch_size, block_size))
    logits = model(idx)
    assert logits.shape == (batch_size, block_size, vocab_size)
    print(f"  GPT forward: {idx.shape} -> {logits.shape}  \u2713")

    print("\n  Testing generate()...")
    context   = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=10)
    assert generated.shape == (1, 11)
    print(f"  generate: {context.shape} -> {generated.shape}  \u2713")

    print("\nAll shape tests passed!")
