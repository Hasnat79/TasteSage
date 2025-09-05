import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from data_loader import NourishRecipeDataset
from models import GPT, GPTConfig

# --------------------------- Constants ---------------------------

DATA_PATH = "../data/combined_recipes_details.json"
BEST_MODEL_PATH = "best_model_nourish.pt"
LOSS_PLOT_PATH = "loss_plot_nourish.png"

# Model architecture
VOCAB_SIZE = 50257
BLOCK_SIZE = 64
NUM_LAYERS = 6
NUM_HEADS = 6
EMBED_DIM = 384
DROPOUT = 0.1
USE_BIAS = True

# Training hyperparameters
MAX_ITERS = 10_000
EVAL_INTERVAL = 500
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
MIN_LR = 5e-4
GRAD_ACCUM_STEPS = 32
WEIGHT_DECAY = 0.1
SEED = 42

# --------------------------- Utility Functions ---------------------------

def get_device():
    if torch.backends.mps.is_available():
        print("Using MPS (Mac GPU) for inference.")
        return "mps"
    
    elif torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference.")
        return "cuda"
    
    print("Using CPU for inference.")
    return "cpu"

def get_dtype(device):
    if device == "mps":
        return "float16"
    return "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"

def evaluate_model_loss(model, dataset, eval_iters, ctx, device):
    model.eval()
    losses = {"train": torch.zeros(eval_iters, device=device), "val": torch.zeros(eval_iters, device=device)}
    
    with torch.inference_mode():
        for split in ['train', 'val']:
            for i in range(eval_iters):
                x, y = dataset.get_batch(split, batch_size=16, block_size=BLOCK_SIZE)
                x, y = x.to(device), y.to(device)
                with ctx:
                    _, loss = model(x, y)
                losses[split][i] = loss.item()
    
    model.train()
    return {k: v.mean() for k, v in losses.items()}

def plot_losses(train_losses, val_losses, interval, path):
    steps = [i * interval for i in range(len(train_losses))]
    train = [loss.cpu().item() for loss in train_losses]
    val = [loss.cpu().item() for loss in val_losses]

    plt.plot(steps, train, label="Train Loss", color='green')
    plt.plot(steps, val, label="Validation Loss", color='red')
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig(path)
    plt.close()

# --------------------------- Training ---------------------------

def train_model():
    torch.manual_seed(SEED)

    # Device and dtype
    device = get_device()
    device_type = "mps" if device == "mps" else ("cuda" if "cuda" in device else "cpu")
    dtype_str = get_dtype(device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_str]
    ctx = nullcontext() if device == "cpu" else torch.amp.autocast(device_type=device_type, dtype=dtype)

    print(f"Using device: {device} ({dtype_str})")

    # Dataset
    dataset = NourishRecipeDataset(DATA_PATH)

    # Model config and instantiation
    config = GPTConfig(
        vocab_size=VOCAB_SIZE,
        block_size=BLOCK_SIZE,
        n_layer=NUM_LAYERS,
        n_head=NUM_HEADS,
        n_embd=EMBED_DIM,
        dropout=DROPOUT,
        bias=USE_BIAS
    )
    model = GPT(config).to(device)

    # Optimizer and learning rate schedulers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=WEIGHT_DECAY,
        eps=1e-9
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, total_iters=WARMUP_STEPS),
            CosineAnnealingLR(optimizer, T_max=MAX_ITERS - WARMUP_STEPS, eta_min=MIN_LR)
        ],
        milestones=[WARMUP_STEPS]
    )

    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda" and dtype_str == "float16"))

    # Training loop
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for step in tqdm(range(MAX_ITERS)):
        if step % EVAL_INTERVAL == 0 and step != 0:
            losses = evaluate_model_loss(model, dataset, eval_iters=EVAL_INTERVAL, ctx=ctx, device=device)
            train_losses.append(losses["train"])
            val_losses.append(losses["val"])
            print(f"[Step {step}] Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"Model checkpoint saved to {BEST_MODEL_PATH}")

        # Training step
        x, y = dataset.get_batch("train", batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
        x, y = x.to(device), y.to(device)

        with ctx:
            _, loss = model(x, y)
            loss = loss / GRAD_ACCUM_STEPS

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == MAX_ITERS:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    plot_losses(train_losses, val_losses, EVAL_INTERVAL, LOSS_PLOT_PATH)
    print(f"Loss plot saved to {LOSS_PLOT_PATH}")

# --------------------------- Entry Point ---------------------------

if __name__ == "__main__":
    train_model()
