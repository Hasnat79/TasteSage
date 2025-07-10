import torch
import tiktoken
from huggingface_hub import hf_hub_download

from models import GPTConfig, GPT

# ----------------------- Constants -----------------------

REPO_ID = "hasnat79/tiny_stories_gpt2_60k_epoch"
MODEL_FILENAME = "best_model_params_60K_EP.pt"
GENERATE_LENGTH = 200
VOCAB_NAME = "gpt2"
DEVICE_PREFERENCE = ["mps", "cuda", "cpu"]

PROMPT = "A little girl went to the woods"


# ----------------------- Utility Functions -----------------------

def get_best_available_device():
    """
    Returns the best available device among MPS, CUDA, or CPU.
    """
    if torch.backends.mps.is_available():
        print("Using MPS (Mac GPU) for inference.")
        return "mps"

    elif torch.cuda.is_available():
        print("Using CUDA for inference.")
        return "cuda"

    else:
        print("Using CPU for inference.")
        return "cpu"

def load_model(config: GPTConfig, device: str) -> GPT:
    """
    Downloads and loads the pre-trained model from Hugging Face Hub.
    """
    print(f"Downloading model weights from Hugging Face Hub ({REPO_ID})...")
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)

    model = GPT(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)

    print(f"Model loaded on {device.upper()}")
    return model

def generate_text(model: GPT, prompt: str, max_tokens: int, device: str) -> str:
    """
    Generates text continuation from the prompt using the GPT model.
    """
    tokenizer = tiktoken.get_encoding(VOCAB_NAME)
    context_tokens = torch.tensor(tokenizer.encode_ordinary(prompt)).unsqueeze(0).to(device)

    output_tokens = model.generate(context_tokens, max_tokens)
    decoded_text = tokenizer.decode(output_tokens.squeeze().tolist())
    return decoded_text

# ----------------------- Main Function -----------------------

def main():
    # Select device
    device = get_best_available_device()
    print(f"Using device: {device.upper()} for inference.")

    # Define model configuration (must match training config)
    config = GPTConfig(
        vocab_size=50257,
        block_size=128,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
        bias=True,
    )

    # Load model
    model = load_model(config, device)

    # Generate text
    print(f"Prompt: {PROMPT!r}")
    output = generate_text(model, prompt=PROMPT, max_tokens=GENERATE_LENGTH, device=device)
    print("\nGenerated Text:\n")
    print(output)

# ----------------------- Entry Point -----------------------

if __name__ == "__main__":
    main()
