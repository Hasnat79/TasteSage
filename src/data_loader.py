import os
import json
import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
import tiktoken

# ===================== Constants & Configuration ===================== #

# Device selection
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DEVICE_TYPE = "mps" if DEVICE == "mps" else ("cuda" if "cuda" in DEVICE else "cpu")

# Tokenizer configuration
TOKENIZER_NAME = "gpt2"
ENCODER = tiktoken.get_encoding(TOKENIZER_NAME)

# Paths and names
TINYSTORIES_HF_NAME = "roneneldan/TinyStories"
TINYSTORIES_DATA_DIR = "./tiny_stories_dataset"
TINYSTORIES_TRAIN_FILE = os.path.join(TINYSTORIES_DATA_DIR, "train.bin")
TINYSTORIES_VAL_FILE = os.path.join(TINYSTORIES_DATA_DIR, "validation.bin")

NOURISH_DATA_PATH = "../data/combined_recipes_details.json"
NOURISH_DATA_DIR = "./nourish_recipe_dataset"
NOURISH_TRAIN_FILE = os.path.join(NOURISH_DATA_DIR, "train.bin")
NOURISH_VAL_FILE = os.path.join(NOURISH_DATA_DIR, "val.bin")

# Tokenization batching
NUM_SHARDS = 1024
NP_DTYPE = np.uint16
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

# ===================== Utility Functions ===================== #

def tokenize_text(example):
    """Tokenize plain text using GPT-2 tokenizer."""
    token_ids = ENCODER.encode_ordinary(example["text"])
    return {"ids": token_ids, "len": len(token_ids)}

# ===================== TinyStories Dataset Class ===================== #

class TinyStoriesDataset:
    def __init__(self):
        self.train_path = TINYSTORIES_TRAIN_FILE
        self.val_path = TINYSTORIES_VAL_FILE
        self.dataset = load_dataset(TINYSTORIES_HF_NAME)
        print(f"Dataset loaded with splits: {self.dataset.keys()}")

        print(f"Sample: {self.dataset['train'][0]['text'][:50]}...\n")
        self._save_tokenized_data()

    def _save_tokenized_data(self):
        """Tokenize and save the full TinyStories dataset as binary .bin files."""
        if os.path.exists(self.train_path) and os.path.exists(self.val_path):
            print("Tokenized files already exist. Skipping tokenization.")
            return

        tokenized = self.dataset.map(
            tokenize_text,
            remove_columns=["text"],
            desc="Tokenizing TinyStories dataset",
            num_proc=8
        )

        os.makedirs(TINYSTORIES_DATA_DIR, exist_ok=True)

        for split, dset in tokenized.items():
            total_tokens = np.sum(dset["len"], dtype=np.uint64)
            filename = os.path.join(TINYSTORIES_DATA_DIR, f"{split}.bin")
            arr = np.memmap(filename, dtype=NP_DTYPE, mode="w+", shape=(total_tokens,))
            idx = 0

            for batch_idx in tqdm(range(NUM_SHARDS), desc=f"Writing {filename}"):
                batch = dset.shard(NUM_SHARDS, batch_idx, contiguous=True).with_format("numpy")
                batch_ids = np.concatenate(batch["ids"])
                arr[idx:idx+len(batch_ids)] = batch_ids
                idx += len(batch_ids)

            arr.flush()
            print(f"{split} split saved with {total_tokens} tokens to {filename}")

    def get_batch(self, split="train", batch_size=32, block_size=128):
        path = self.train_path if split == "train" else self.val_path
        data = np.memmap(path, dtype=NP_DTYPE, mode="r")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        return self._to_device(x, y)

    def _to_device(self, x, y):
        if DEVICE_TYPE == "cuda":
            return x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
        return x.to(DEVICE), y.to(DEVICE)

# ===================== Nourish Recipe Dataset Class ===================== #

class NourishRecipeDataset:
    def __init__(self, data_path=NOURISH_DATA_PATH):
        # Define special tokens for recipe schema
        # self.special_tokens = {
        #     '<bos>': 50257,     # Beginning of sequence
        #     '<eos>': 50258,     # End of sequence  
        #     '<recipe_start>': 50259,
        #     '<recipe_end>': 50260,
        #     '<q_start>': 50261,
        #     '<q_end>': 50262,
        #     '<ans_start>': 50263,
        #     '<ans_end>': 50264
        # }# unused
        
        # --------------- Special tokens for GPT-2 tokenizer --------------- #
        special_tokens = ['<bos>', '<eos>', '<recipe_start>', '<recipe_end>', 
                  '<q_start>', '<q_end>', '<ans_start>', '<ans_end>']

        # Register with tokenizer
        ENCODER.add_special_tokens({'additional_special_tokens': special_tokens})

        # Update special token dict with  actual IDs
        self.special_tokens = {tok: ENCODER.convert_tokens_to_ids(tok) for tok in special_tokens}
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}

        # Paths for train and validation data

        self.train_path = NOURISH_TRAIN_FILE
        self.val_path = NOURISH_VAL_FILE
        
        # Create reverse mapping for decoding
        self.id_to_token = {v: k for k, v in self.special_tokens.items()} # unused


        with open(data_path, "r") as f:
            raw_data = json.load(f)
        
        self.dataset = [r for r in raw_data if self._is_valid(r)]
        print(f"Loaded {len(self.dataset)} valid recipes.")

        self.text_data = self._format_recipes()
        self.tokenized_data = [ENCODER.encode_ordinary(t) for t in tqdm(self.text_data, desc="Tokenizing")]
        self._split_data()
        self._save_binary_files()

    def _is_valid(self, recipe):
        required = ["title", "instructions", "safe_for_diabetes"]
        return all(field in recipe and recipe[field] for field in required)

    def _format_recipes(self):
        texts = []
        for recipe in self.dataset:
            
            # Format instructions as a single string
            if isinstance(recipe['instructions'], list):
                instructions = ' '.join(recipe['instructions'])
            else:
                instructions = str(recipe['instructions'])
                
            text = (
                f"<bos> "
                f"<recipe_start> "
                f"Recipe Name: {recipe['title']} "
                f"Instructions: {instructions} "
                f"<recipe_end> "
                f"<q_start> Is this recipe safe for diabetes? <q_end> "
                f"<ans_start> {recipe['safe_for_diabetes']}. <ans_end> " # first phase
                f"<eos> "
            )
            
            texts.append(text)
        return texts

    def _split_data(self):
        np.random.seed(RANDOM_SEED)
        indices = np.random.permutation(len(self.tokenized_data))
        train_size = int(len(indices) * TRAIN_RATIO)
        self.train_data = [self.tokenized_data[i] for i in indices[:train_size]]
        self.val_data = [self.tokenized_data[i] for i in indices[train_size:]]
        print(f"Train: {len(self.train_data)}, Val: {len(self.val_data)}")

    def _save_binary_files(self):
        os.makedirs(NOURISH_DATA_DIR, exist_ok=True)
        for split, data in [("train", self.train_data), ("val", self.val_data)]:
            flat_ids = np.array([id for seq in data for id in seq], dtype=NP_DTYPE)
            path = os.path.join(NOURISH_DATA_DIR, f"{split}.bin")
            flat_ids.tofile(path)
            print(f"Saved {len(flat_ids)} tokens to {path}")

    def get_batch(self, split="train", batch_size=32, block_size=128):
        path = self.train_path if split == "train" else self.val_path
        data = np.memmap(path, dtype=NP_DTYPE, mode="r")
        if len(data) <= block_size:
            raise ValueError(f"Data too short for block size: {len(data)} < {block_size}")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        return self._to_device(x, y)

    def _to_device(self, x, y):
        if DEVICE_TYPE == "cuda":
            return x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
        return x.to(DEVICE), y.to(DEVICE)

# ===================== Main (Test Entry Point) ===================== #

if __name__ == "__main__":
    print(f"Using device: {DEVICE.upper()}")
    
    # # Test TinyStories
    # print("\n--- Testing TinyStoriesDataset ---")
    # tiny_dataset = TinyStoriesDataset()
    # x, y = tiny_dataset.get_batch("train", batch_size=4, block_size=64)
    # print(f"TinyStories batch - x: {x.shape}, y: {y.shape}")
    
    # Test NourishRecipes
    print("\n--- Testing NourishRecipeDataset ---")
    nourish_dataset = NourishRecipeDataset()
    x, y = nourish_dataset.get_batch("train", batch_size=4, block_size=64)
    print(f"Nourish batch - x: {x.shape}, y: {y.shape}")
    
    sample_decoded = ENCODER.decode(x[0].tolist())
    print("\nSample decoded text:\n", sample_decoded)
    print("\nDone with data loading tests.")