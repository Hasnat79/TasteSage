from data_loader import YelpFoodBusinessReviewsDataset
from models import GPTConfig, GPT
import torch
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR,SequentialLR, CosineAnnealingLR
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def estimate_loss(model,dataset,eval_iters=500, ctx=nullcontext()):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'test']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = dataset.get_batch(split, batch_size=16, block_size=128)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    dataset = YelpFoodBusinessReviewsDataset(data_dir="../data/yelp_food_business_review_dataset") 

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

    PREV_BEST_MODEL_PATH = 'taste_sage_best_model_params_20k.pt'
    BEST_MODEL_PATH = 'taste_sage_best_model_params_40k.pt'
    LOSS_FIG_PATH = 'taste_sage_loss_plot_40k_start_from_20k_ckpt.png'

    # if model path available, load the model 
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            model.load_state_dict(torch.load(PREV_BEST_MODEL_PATH, map_location=torch.device('cuda')))
            print("Loaded model with bfloat16 support.")
        else:
            model.load_state_dict(torch.load(PREV_BEST_MODEL_PATH, map_location=torch.device('cpu')))
            print("Loaded model without bfloat16 support.")
    learning_rate = 1e-4 #more stable training, earlier 1e-4
    # max_iters = 60000 #increase from 25000
    max_iters = 20000 #increase from 25000
    warmup_steps = 1000 #smoother initial train, earlier 100
    min_lr = 5e-4 #lower rate, earlier 5e-4
    eval_iters = 500 # increased from 100
    batch_size = 16 # changed from 16, better gradient estimate
    block_size = 128 #changed from 64, capture longer range dependencies

    gradient_accumulation_steps = 32 # reduced from 50

    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

    # note: float16 data type will automatically use a GradScaler

    # How to use autocast https://wandb.ai/wandb_fc/tips/reports/How-To-Use-Autocast-in-PyTorch--VmlldzoyMTk4NTky
    #dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    torch.set_default_device(device)
    torch.manual_seed(42)

    # optimizers and schedulers
    ##PUT IN WEIGHT DECAY, CHANGED BETA2 to 0.95
    optimizer =  torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9) #weight decay for regularization

    scheduler_warmup = LinearLR(optimizer, total_iters = warmup_steps) #Implement linear warmup
    scheduler_decay = CosineAnnealingLR(optimizer,T_max = max_iters - warmup_steps, eta_min = min_lr) #Implement lr decay
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps]) #Switching from warmup to decay

    # https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # training

    best_val_loss = float('inf')
    best_model_params_path = BEST_MODEL_PATH
    train_loss_list, validation_loss_list = [], []


    # In your training loop
    for epoch in tqdm(range(max_iters)):
        if epoch % eval_iters == 0 and epoch != 0:
            # Ensure estimate_loss uses the correct device
            losses = estimate_loss(model, dataset, eval_iters=eval_iters, ctx=ctx)
            print(f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")
            print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
            train_loss_list += [losses['train']]
            validation_loss_list += [losses['test']]

            if losses['test'] < best_val_loss:
                best_val_loss = losses['test']
                torch.save(model.state_dict(), best_model_params_path)

        # Ensure X and y are on the correct device
        X, y = dataset.get_batch("train", batch_size=batch_size, block_size=block_size)
        X, y = X.to(device), y.to(device)

        with ctx:
            logits, loss = model(X, y)
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()

        if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    # plot SLM loss func

    train_loss_list_converted = [i.cpu().detach() for i in train_loss_list]
    validation_loss_list_converted = [i.cpu().detach() for i in validation_loss_list]

    plt.plot(train_loss_list_converted, 'g', label='train_loss')
    plt.plot(validation_loss_list_converted, 'r', label='validation_loss')
    plt.xlabel("Steps - Every 500 epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(LOSS_FIG_PATH)
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Best model parameters saved to {best_model_params_path}")