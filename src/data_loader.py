import os
import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_dataset
import tiktoken

# ------------------------------ Constants ------------------------------

DATASET_NAME = "roneneldan/TinyStories"
TOKENIZER_NAME = "gpt2"
DATA_DIR = "./tiny_stories_dataset"
TRAIN_FILE = os.path.join(DATA_DIR, "train.bin")
VAL_FILE = os.path.join(DATA_DIR, "validation.bin")
NUM_SHARDS = 1024
DTYPE = np.uint16  # GPT-2 vocab size < 65536

# ------------------------------ Device Detection ------------------------------

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

device_type = "mps" if device == "mps" else ("cuda" if "cuda" in device else "cpu")

# ------------------------------ Tokenization ------------------------------

def tokenize_example(example):
    """
    Tokenizes a single dataset example using GPT-2 tokenizer.
    """
    tokenizer = tiktoken.get_encoding(TOKENIZER_NAME)
    token_ids = tokenizer.encode_ordinary(example["text"])
    return {"ids": token_ids, "len": len(token_ids)}

# ------------------------------ Dataset Class ------------------------------

class TinyStoriesDataset:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.train_path = TRAIN_FILE
        self.val_path = VAL_FILE

        print("Loading dataset...")
        self.raw_dataset = load_dataset(DATASET_NAME)
        print(f"Splits: {self.raw_dataset.keys()}")

        # Preview one example
        example_text = self.raw_dataset["train"][0]["text"]
        print(f"Example from dataset:\n{example_text[:50]}...\n")

        self._prepare_tokenized_data()

    def _prepare_tokenized_data(self):
        """
        Tokenizes and saves the dataset to disk if not already processed.
        """
        if os.path.exists(self.train_path) and os.path.exists(self.val_path):
            print("Found cached dataset. Skipping processing.")
            return

        os.makedirs(self.data_dir, exist_ok=True)

        print("Tokenizing dataset...")
        tokenized = self.raw_dataset.map(
            tokenize_example,
            remove_columns=["text"],
            desc="Tokenizing splits",
            num_proc=8
        )

        for split, dset in tokenized.items():
            total_tokens = np.sum(dset["len"], dtype=np.uint64)
            output_path = os.path.join(self.data_dir, f"{split}.bin")

            print(f"Writing {split} split: {len(dset)} samples, {total_tokens} tokens.")
            arr = np.memmap(output_path, dtype=DTYPE, mode="w+", shape=(total_tokens,))
            idx = 0

            for shard_idx in tqdm(range(NUM_SHARDS), desc=f"Writing {split}.bin"):
                shard = dset.shard(num_shards=NUM_SHARDS, index=shard_idx, contiguous=True).with_format("numpy")
                shard_ids = np.concatenate(shard["ids"])
                arr[idx : idx + len(shard_ids)] = shard_ids
                idx += len(shard_ids)

            arr.flush()
            print(f"Saved {split} split to {output_path}")

    def get_batch(self, split="train", batch_size=32, block_size=128):
        """
        Returns a (x, y) batch pair of token sequences.
        """
        filepath = self.train_path if split == "train" else self.val_path
        data = np.memmap(filepath, dtype=DTYPE, mode="r")

        indices = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in indices])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in indices])

        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        return x, y

# ------------------------------ Main (Testing) ------------------------------

if __name__ == "__main__":
    dataset = TinyStoriesDataset()

    # # Sample batch
    # x, y = dataset.get_batch(split="train", batch_size=2, block_size=64)
    # print(f"x shape: {x.shape}, y shape: {y.shape}")
    # print(f"x sample:\n{x[0]}")
    # print(f"y sample:\n{y[0]}")
