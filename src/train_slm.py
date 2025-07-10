import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

from data_loader import TinyStoriesDataset
from models import GPTConfig, GPT, load_config


# ------------------------ Utility Functions ------------------------

def evaluate_model_loss(model, dataset, eval_steps=500, ctx=nullcontext(), device='cpu'):
    """Estimate train and validation loss over a number of evaluation steps."""
    model.eval()
    loss_dict = {}
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_steps, device=device)
            for i in range(eval_steps):
                inputs, targets = dataset.get_batch(split, batch_size=16, block_size=128)
                inputs, targets = inputs.to(device), targets.to(device)
                with ctx:
                    _, loss = model(inputs, targets)
                losses[i] = loss.item()
            loss_dict[split] = losses.mean()
    model.train()
    return loss_dict

def plot_loss_curves(train_losses, val_losses, eval_interval, output_path):
    """Plot training and validation loss curves."""
    x = [i * eval_interval for i in range(len(train_losses))]
    plt.plot(x, train_losses, 'g', label='Train Loss')
    plt.plot(x, val_losses, 'r', label='Validation Loss')
    plt.xlabel("Training Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

# ------------------------ Training Function ------------------------

def train_gpt_model():
    # Load hyperparameters and config
    cfg = load_config("config/model_config.yaml")

    # ------------------- Device Setup -------------------
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Mac GPU).")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA (GPU).")
    else:
        device = "cpu"
        print("Using CPU.")

    device_type = "mps" if device == "mps" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = "float16" if device_type == "mps" else (
        "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    )
    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch_dtype)

    # ------------------- Dataset & Model -------------------
    torch.manual_seed(cfg.seed)
    dataset = TinyStoriesDataset()

    model_config = GPTConfig(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.num_layers,
        n_head=cfg.num_heads,
        n_embd=cfg.embedding_dim,
        dropout=cfg.dropout,
        bias=True,
    )
    model = GPT(model_config).to(device)

    # ------------------- Optimizer & Scheduler -------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        eps=1e-9
    )
    scheduler_warmup = LinearLR(optimizer, total_iters=cfg.warmup_steps)
    scheduler_decay = CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_iters - cfg.warmup_steps,
        eta_min=cfg.min_lr
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[cfg.warmup_steps]
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda" and dtype == "float16"))

    # ------------------- Training Loop -------------------
    best_val_loss = float("inf")
    train_loss_log, val_loss_log = [], []

    for step in tqdm(range(cfg.max_iters)):
        if step % cfg.eval_interval == 0 and step != 0:
            loss_dict = evaluate_model_loss(model, dataset, eval_steps=cfg.eval_interval, ctx=ctx, device=device)
            
            train_loss_log.append(loss_dict["train"].cpu().item())
            val_loss_log.append(loss_dict["val"].cpu().item()) 
            
            print(f"Step {step}: Train Loss = {loss_dict['train']:.4f}, Val Loss = {loss_dict['val']:.4f}")
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

            if loss_dict["val"] < best_val_loss:
                best_val_loss = loss_dict["val"]
                torch.save(model.state_dict(), cfg.best_model_path)

        # Training step
        inputs, targets = dataset.get_batch("train", batch_size=cfg.batch_size, block_size=cfg.block_size)
        inputs, targets = inputs.to(device), targets.to(device)

        with ctx:
            _, loss = model(inputs, targets)
            loss = loss / cfg.grad_accum_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == cfg.max_iters:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

    # ------------------- Finalization -------------------
    print(f"Training completed. Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {cfg.best_model_path}")
    plot_loss_curves(train_loss_log, val_loss_log, eval_interval=cfg.eval_interval, output_path=cfg.loss_plot_path)

# ------------------------ Entry Point ------------------------

if __name__ == "__main__":
    train_gpt_model()
