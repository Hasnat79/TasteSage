import torch
import tiktoken
from models import GPTConfig, GPT
# from huggingface_hub import hf_hub_download  # Optional: if loading from Hugging Face

# ----------------------------- Constants -----------------------------

# Model configuration
MODEL_CONFIG = {
    "vocab_size": 50257,
    "block_size": 64,
    "n_layer": 6,
    "n_head": 6,
    "n_embd": 384,
    "dropout": 0.1,
    "bias": True,
}

# Inference prompt
PROMPT_TEXT = """
<bos> 
<recipe_start> 
Recipe Name: Mini Lime Cheesecake Bites 
Instructions: Preheat oven to 275 F. Line a mini muffin pan with 24 muffin liners. 
    "For the crust: In a bowl add oat flour, oats, baking powder, cinnamon and salt. 
    "Stir until combined. 
    "Add in the applesauce, maple syrup and vanilla; mix well. 
    "Scoop about 1 heaping teaspoon of batter evenly into each muffin cup and push down firmly with fingers to smooth out. 
    "I found that wetting your fingers makes the base not stick to them. 
    "For the cheesecake filling: In a medium sized bowl add cream cheese and sugar; beat until creamy. 
    "Add vanilla extract, egg and salt; beat until combined. 
    "Beat in yogurt and lime juice until creamy. 
    "Fold in lime zest. 
    "Evenly pour cheesecake batter into muffin cups. 
    "NOTE: You will have leftover batter. 
    "Bake, rotating pans halfway through, until filling is set. 
    "This will take about 15 minutes. 
    "Remove pan from the oven and set on a wire rack until cooled. 
    "Chill in the refrigerator for at least 4 hours, or up to overnight, before serving."
<recipe_end>
<q_start> Is this recipe safe for diabetes?
<q_end>
<ans_start> 
"""

# Generation settings
GEN_MAX_TOKENS = 300
TOKENIZER_NAME = "gpt2"

# Paths (use one of these)
LOCAL_MODEL_PATH = "/Users/hasnatmdabdullah/Documents/Developer/TasteSage/src/best_model_params_10000_EP.pt"
# HUGGINGFACE_REPO_ID = "hasnat79/tiny_stories_gpt2_60k_epoch"
# HUGGINGFACE_MODEL_FILENAME = "best_model_params_60K_EP.pt"

# ----------------------------- Utility Functions -----------------------------

def select_device():
    """Selects the best available device for inference."""
    if torch.backends.mps.is_available():
        print("Using MPS (Mac GPU) for inference.")
        return "mps"
    
    elif torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference.")
        return "cuda"
    
    print("Using CPU for inference.")
    return "cpu"

def load_model(config: dict, weights_path: str, device: str) -> GPT:
    """Initializes the model and loads pre-trained weights."""
    model_config = GPTConfig(**config)
    model = GPT(model_config)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    return model

def generate_response(model: GPT, prompt: str, max_tokens: int, device: str) -> str:
    """Encodes the prompt and generates text using the model."""
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    input_ids = torch.tensor(tokenizer.encode_ordinary(prompt)).unsqueeze(0).to(device)
    output_ids = model.generate(input_ids, max_tokens)
    return tokenizer.decode(output_ids.squeeze().tolist())

# ----------------------------- Main Entry -----------------------------

def main():
    device = select_device()
    print(f"Using device: {device.upper()}")

    print("Initializing model...")
    model = load_model(MODEL_CONFIG, LOCAL_MODEL_PATH, device)

    print("Generating response...\n")
    generated_text = generate_response(model, PROMPT_TEXT, GEN_MAX_TOKENS, device)
    print(generated_text)

# ----------------------------- Entry Point -----------------------------

if __name__ == "__main__":
    main()
