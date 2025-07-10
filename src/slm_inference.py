from models import GPTConfig, GPT
import tiktoken 
import torch
from huggingface_hub import hf_hub_download

if __name__ == "__main__":
    config = GPTConfig(
    vocab_size=50257,     # use the tokenizer's vocab size
    block_size=64,       # or whatever context size you're training with
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    bias=True
)
    model = GPT(config)  # re-create the model with same config
    
    # Detect best available device: MPS (Mac GPU) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Mac GPU) for inference.")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA for inference.")
    else:
        device = "cpu"
        print("Using CPU for inference.")
    
    best_model_params_path = "/Users/hasnatmdabdullah/Documents/Developer/TasteSage/src/best_model_params_10000_EP.pt"
    # Download the model from HuggingFace
    # best_model_params_path = hf_hub_download(repo_id="hasnat79/tiny_stories_gpt2_60k_epoch", 
                                            #  filename="best_model_params_60K_EP.pt")
    model.load_state_dict(torch.load(best_model_params_path, map_location=torch.device(device))) # load best model states
    model.to(device)  # Ensure model is on the correct device

    sentence = """
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
    
    
    
   
    enc = tiktoken.get_encoding("gpt2")
    context = (torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(dim = 0)).to(device)
    y = model.generate(context, 300)
    print(enc.decode(y.squeeze().tolist()))

    