from models import GPTConfig, GPT
import tiktoken 
import torch

if __name__ == "__main__":
    config = GPTConfig(
    vocab_size=50257,     # use the tokenizer's vocab size
    block_size=128,       # or whatever context size you're training with
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)
    model = GPT(config)  # re-create the model with same config
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    best_model_params_path = "/scratch/user/hasnat.md.abdullah/TasteSage/vizuara/best_model_params_20K_EP_.pt"
    model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device))) # load best model states

    sentence = "A little girl went to the woods"
    enc = tiktoken.get_encoding("gpt2")
    context = (torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim = 0))
    y = model.generate(context, 200)
    print(enc.decode(y.squeeze().tolist()))