"""
train_gpt.py  —  Introduction to AI | Final Assignment
============================================================
Training script for Task 2: Character-Level GPT.

Usage:
    python train_gpt.py

Before running:
    1. Implement AttentionBlock in models.py        (Task 1)
    2. Implement Transformer_Block.forward          (Task 2A)
    3. Implement GPT.forward                        (Task 2B)
    4. Place your text corpus in this folder and
       set CORPUS_FILE below (min 500,000 characters)

Hyperparameters to experiment with:
    BLOCK_SIZE    — context window length (characters)
    LAYER_SIZE    — embedding / hidden dimension
    N_LAYERS      — number of stacked Transformer blocks
    LEARNING_RATE — Adam optimizer learning rate
    MAX_ITERS     — total number of training steps
    BATCH_SIZE    — sequences per training batch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt_model import GPT

# =============================================================================
#  Configuration
# =============================================================================

CORPUS_FILE   = 'input.txt'   # path to your text corpus (Task 2 corpus choice)

BLOCK_SIZE    = 64            # context window in characters
LAYER_SIZE    = 64            # embedding dimension
N_LAYERS      = 2             # number of Transformer blocks
LEARNING_RATE = 3e-4          # Adam learning rate
MAX_ITERS     = 30000         # total training steps
BATCH_SIZE    = 32            # sequences per training batch
EVAL_INTERVAL = 300           # print loss every N steps
EVAL_ITERS    = 50            # steps averaged when estimating loss
GENERATE_LEN  = 500           # characters to generate after training

# =============================================================================
#  Reproducibility
# =============================================================================

torch.manual_seed(42)

# =============================================================================
#  Load and prepare data
# =============================================================================

print(f"\nLoading corpus: {CORPUS_FILE}")
with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"  Total characters : {len(text):,}")

# Build character-level vocabulary
chars      = sorted(set(text))
vocab_size = len(chars)
print(f"  Vocabulary size  : {vocab_size} unique characters")
print(f"  Characters       : {repr(''.join(chars[:40]))}{'...' if vocab_size > 40 else ''}")

# Character <-> index mappings
stoi   = {ch: i for i, ch in enumerate(chars)}
itos   = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train / validation split  (90% / 10%)
data       = torch.tensor(encode(text), dtype=torch.long)
split      = int(0.9 * len(data))
train_data = data[:split]
val_data   = data[split:]
print(f"  Training tokens  : {len(train_data):,}")
print(f"  Validation tokens: {len(val_data):,}")

# =============================================================================
#  Batch sampling
# =============================================================================

def get_batch(split: str):
    """
    Sample a random batch of (input, target) sequence pairs.

    Returns:
        x: LongTensor (BATCH_SIZE, BLOCK_SIZE)  — input sequences
        y: LongTensor (BATCH_SIZE, BLOCK_SIZE)  — target sequences (x shifted by 1)
    """
    d  = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - BLOCK_SIZE, (BATCH_SIZE,))
    x  = torch.stack([d[i     : i + BLOCK_SIZE    ] for i in ix])
    y  = torch.stack([d[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x, y

# =============================================================================
#  Loss estimation
# =============================================================================

@torch.no_grad()
def estimate_loss(model: nn.Module) -> dict:
    """Estimate mean cross-entropy loss on train and val splits."""
    model.eval()
    results = {}
    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y    = get_batch(split)
            logits  = model(X)
            B, T, C = logits.shape
            loss    = F.cross_entropy(
                logits.view(B * T, C),
                Y.view(B * T)
            )
            losses[k] = loss.item()
        results[split] = losses.mean().item()
    model.train()
    return results

# =============================================================================
#  Build model
# =============================================================================

model = GPT(
    vocab_size = vocab_size,
    block_size = BLOCK_SIZE,
    layer_size = LAYER_SIZE,
    n_layers   = N_LAYERS,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters : {n_params:,}")
print(f"Training for     : {MAX_ITERS} steps\n")

# =============================================================================
#  Optimizer
# =============================================================================

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =============================================================================
#  Training loop
# =============================================================================

print(f"{'Step':>6}  {'Train Loss':>12}  {'Val Loss':>10}")
print("-" * 36)

for step in range(MAX_ITERS):

    # Estimate and print loss periodically
    if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
        losses = estimate_loss(model)
        print(f"{step:6d}  {losses['train']:12.4f}  {losses['val']:10.4f}")

    # Forward pass
    X, Y    = get_batch('train')
    logits  = model(X)
    B, T, C = logits.shape
    loss    = F.cross_entropy(
        logits.view(B * T, C),
        Y.view(B * T)
    )

    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# =============================================================================
#  Generate a text sample
# =============================================================================

print("\n" + "=" * 60)
print(f"GENERATED SAMPLE ({GENERATE_LEN} characters):")
print("=" * 60)

model.eval()
seed_char  = text[0]
context    = torch.tensor([[stoi[seed_char]]], dtype=torch.long)
generated  = model.generate(context, max_new_tokens=GENERATE_LEN)
print(decode(generated[0].tolist()))
print("=" * 60)

print("\nTraining complete.")
print("Include the loss table above and the generated sample in your analysis.pdf.")
