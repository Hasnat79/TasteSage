from data_loader import TinyStoriesDataset
from models import GPTConfig, GPT
import torch
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR,SequentialLR, CosineAnnealingLR
from tqdm.auto import tqdm


def estimate_loss(model,dataset,eval_iters=500, ctx=nullcontext(), device='cpu'):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters, device=device)
            for k in range(eval_iters):
                X, Y = dataset.get_batch(split, batch_size=16, block_size=128)
                X, Y = X.to(device), Y.to(device)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":
    # Detect best available device: MPS (Mac GPU) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print("MPS (Mac GPU) is available and will be used.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available and will be used.")
    else:
        device = "cpu"
        print("Using CPU.")
    
    print(f"Using device: {device}")
    dataset = TinyStoriesDataset()

    config = GPTConfig(
    vocab_size=50257,     # use the tokenizer's vocab size
    block_size=128,       # or whatever context size you're training with
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)
    model = GPT(config) 
    model.to(device)
    # training config

    


    learning_rate = 1e-4 #more stable training, earlier 1e-4
    max_iters = 101 #increase from 25000
    warmup_steps = 2 #smoother initial train, earlier 100
    min_lr = 5e-4 #lower rate, earlier 5e-4
    eval_iters = 25# increased from 100
    batch_size = 32 # changed from 16, better gradient estimate
    block_size = 128 #changed from 64, capture longer range dependencies

    gradient_accumulation_steps = 32 # reduced from 50

    BEST_MODEL_PATH = f'best_model_params_{max_iters}_EP.pth'
    LOSS_FIG_PATH = f'loss_plot_{max_iters}_EP.png'

    device_type = 'mps' if device == 'mps' else ('cuda' if 'cuda' in device else 'cpu') # for later use in torch.autocast

    # note: float16 data type will automatically use a GradScaler
    # MPS doesn't support bfloat16, so use float16 for MPS and bfloat16/float16 for CUDA
    if device == 'mps':
        dtype = 'float16'  # MPS only supports float16
    else:
        dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    
    # How to use autocast https://wandb.ai/wandb_fc/tips/reports/How-To-Use-Autocast-in-PyTorch--VmlldzoyMTk4NTky
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Don't set default device for MPS as it can cause allocation issues
    # torch.set_default_device(device)  # Commented out to fix MPS issues
    torch.manual_seed(42)

    # optimizers and schedulers
    ##PUT IN WEIGHT DECAY, CHANGED BETA2 to 0.95
    optimizer =  torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9) #weight decay for regularization

    scheduler_warmup = LinearLR(optimizer, total_iters = warmup_steps) #Implement linear warmup
    scheduler_decay = CosineAnnealingLR(optimizer,T_max = max_iters - warmup_steps, eta_min = min_lr) #Implement lr decay
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps]) #Switching from warmup to decay

    # https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
    # Use GradScaler only for CUDA with float16, MPS doesn't need GradScaler
    if device_type == 'cuda':
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    else:
        scaler = None  # MPS and CPU don't use GradScaler

    # training

    best_val_loss = float('inf')
    best_model_params_path = BEST_MODEL_PATH
    train_loss_list, validation_loss_list = [], []


    # In your training loop
    for epoch in tqdm(range(max_iters)):
        if epoch % eval_iters == 0 and epoch != 0:
            # Ensure estimate_loss uses the correct device
            losses = estimate_loss(model, dataset, eval_iters=eval_iters, ctx=ctx, device=device)
            print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
            train_loss_list += [losses['train']]
            validation_loss_list += [losses['val']]

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), best_model_params_path)

        # Ensure X and y are on the correct device
        X, y = dataset.get_batch("train", batch_size=batch_size, block_size=block_size)
        X, y = X.to(device), y.to(device)

        with ctx:
            logits, loss = model(X, y)
            loss = loss / gradient_accumulation_steps
            
            # Handle gradient scaling based on device
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    # plot SLM loss func
    import matplotlib.pyplot as plt
    print(f"len(train_loss_list) = {len(train_loss_list)}, len(validation_loss_list) = {len(validation_loss_list)}")
    train_loss_list_converted = [i.cpu().detach() for i in train_loss_list]
    validation_loss_list_converted = [i.cpu().detach() for i in validation_loss_list]

    # Create x-axis values that show the actual epoch numbers when losses were evaluated
    x_values = [i * eval_iters for i in range(len(train_loss_list_converted))]

    plt.plot(x_values, train_loss_list_converted, 'g', label='train_loss')
    plt.plot(x_values, validation_loss_list_converted, 'r', label='validation_loss')
    plt.xlabel("Total Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(LOSS_FIG_PATH)
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Best model parameters saved to {best_model_params_path}")

    