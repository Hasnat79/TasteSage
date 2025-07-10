from datasets import load_dataset
import tiktoken 
import os 
import numpy as np
from tqdm.auto import tqdm
import torch
import json 


# Detect best available device: MPS (Mac GPU) > CUDA > CPU
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

device_type = 'mps' if device == 'mps' else ('cuda' if 'cuda' in device else 'cpu')

def process(example):
    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(example['text'])  # encode_ordinary ignores any special tokens
    out = {'ids': ids, 'len': len(ids)}
    return out
class TinyStoriesDataset: 
    def __init__(self):
        self.ds = load_dataset("roneneldan/TinyStories")
        print(f"Loaded dataset with splits: {self.ds.keys()}")
        print(type(self.ds))
        for i in range(len(self.ds['train'])):
            print(f"Example {i}: {self.ds['train'][i]['text'][:50]}...")
            break
        self.train_path = "./tiny_stories_dataset/train.bin"
        self.validation_path = "./tiny_stories_dataset/validation.bin"
        self.save_to_disk("./tiny_stories_dataset")

    def save_to_disk(self, path):
        if not os.path.exists(self.train_path):
            self.tokenized = self.ds.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=8,
        )
            # concatenate all the ids in each dataset into one large file we can use for training
            for split, dset in self.tokenized.items():
                arr_len = np.sum(dset['len'], dtype=np.uint64)
                print(f'Processing split: {split}, total samples: {len(dset)}, total tokens: {arr_len}')
                filename = f'{path}/{split}.bin'
                if not os.path.exists(path):
                    os.makedirs(path)
                dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
                arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
                total_batches = 1024

                idx = 0
                for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                    # Batch together samples for faster write
                    batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                    arr_batch = np.concatenate(batch['ids'])
                    # Write into mmap
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()
    
    def get_batch(self,split, batch_size=32, block_size=128): 
        if split == 'train':
            data = np.memmap(self.train_path, dtype=np.uint16, mode='r')
        else:
            data = np.memmap(self.validation_path, dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


class NourishRecipeDataset:
    def __init__(self, data_path):
        # Define special tokens for recipe schema
        self.special_tokens = {
            '<bos>': 50257,     # Beginning of sequence
            '<eos>': 50258,     # End of sequence  
            '<recipe_start>': 50259,
            '<recipe_end>': 50260,
            '<q_start>': 50261,
            '<q_end>': 50262,
            '<ans_start>': 50263,
            '<ans_end>': 50264
        }# unused
        
        # Create reverse mapping for decoding
        self.id_to_token = {v: k for k, v in self.special_tokens.items()} # unused
        
        with open(data_path, 'r') as f:
            self.ds = json.load(f)
            print(f"Loaded dataset with {len(self.ds)} recipes.")
            
        # Filter out recipes with missing data
        self.ds = [recipe for recipe in self.ds if self._is_valid_recipe(recipe)]
        print(f"After filtering: {len(self.ds)} valid recipes.")
        
        # Process recipes into LLM format
        self.processed_texts = self._process_recipes()
        print(f"Processed {len(self.processed_texts)} recipe texts.")
        
        # Tokenize the processed texts
        self._tokenize_dataset()
        
        # Create train/val split (80/20)
        self._create_train_val_split()
        
        # Save tokenized data to disk
        self.train_path = "./nourish_recipe_dataset/train.bin"
        self.validation_path = "./nourish_recipe_dataset/val.bin"
        self._save_to_disk()
    
    def _is_valid_recipe(self, recipe):
        """Check if a recipe has all required fields"""
        required_fields = ['title', 'instructions', 'safe_for_diabetes']
        return all(field in recipe and recipe[field] for field in required_fields)
    
    def _process_recipes(self):
        """Convert recipes to LLM training format using special tokens"""
        processed_texts = []
        
        for recipe in self.ds:
            # Format instructions as a single string
            if isinstance(recipe['instructions'], list):
                instructions = ' '.join(recipe['instructions'])
            else:
                instructions = str(recipe['instructions'])
            
            # Create the LLM input format with special tokens as placeholders
            # We'll replace these with actual token IDs during tokenization
            formatted_text = (
                f"<bos> "
                f"<recipe_start> "
                f"Recipe Name: {recipe['title']} "
                f"Instructions: {instructions} "
                f"<recipe_end> "
                f"<q_start> Is this recipe safe for diabetes? <q_end> "
                f"<ans_start> {recipe['safe_for_diabetes']}. <ans_end> " # first phase
                f"<eos> "
            )
            
            processed_texts.append(formatted_text)
        
        return processed_texts
    
    def _tokenize_dataset(self):
        """Tokenize all processed texts"""
        enc = tiktoken.get_encoding("gpt2")
        self.tokenized_data = []
        
        print("Tokenizing recipes...")
        for text in tqdm(self.processed_texts, desc="Tokenizing"):
            ids = enc.encode_ordinary(text)
            self.tokenized_data.append(ids)
    
    def _create_train_val_split(self, train_ratio=0.8):
        """Create train/validation split"""
        np.random.seed(42)  # For reproducible splits
        n_samples = len(self.tokenized_data)
        indices = np.random.permutation(n_samples)
        
        train_size = int(n_samples * train_ratio)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        self.train_data = [self.tokenized_data[i] for i in train_indices]
        self.val_data = [self.tokenized_data[i] for i in val_indices]
        
        print(f"Train samples: {len(self.train_data)}, Val samples: {len(self.val_data)}")
    
    def _save_to_disk(self):
        """Save tokenized data to binary files"""
        os.makedirs("./nourish_recipe_dataset", exist_ok=True)
        
        for split_name, data in [("train", self.train_data), ("val", self.val_data)]:
            # Concatenate all tokenized sequences
            all_ids = []
            for ids in data:
                all_ids.extend(ids)
            
            # Convert to numpy array and save
            arr = np.array(all_ids, dtype=np.uint16)
            filename = f"./nourish_recipe_dataset/{split_name}.bin"
            arr.tofile(filename)
            
            print(f"Saved {len(arr)} tokens to {filename}")
    
    def get_batch(self, split, batch_size=32, block_size=128):
        """Get a batch of data for training"""
        if split == 'train':
            data = np.memmap(self.train_path, dtype=np.uint16, mode='r')
        else:
            data = np.memmap(self.validation_path, dtype=np.uint16, mode='r')
        
        # Ensure we don't go out of bounds
        max_start = len(data) - block_size
        if max_start <= 0:
            raise ValueError(f"Dataset too small. Data length: {len(data)}, block_size: {block_size}")
        
        ix = torch.randint(max_start, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        
        return x, y


if __name__ == "__main__":
    # Test the NourishRecipeDataset
    dataset = NourishRecipeDataset(data_path="../data/combined_recipes_details.json")
    
    # Test getting a batch
    try:
        x, y = dataset.get_batch('train', batch_size=4, block_size=64)
        print(f"Training batch - x shape: {x.shape}, y shape: {y.shape}")
        
        x_val, y_val = dataset.get_batch('val', batch_size=4, block_size=64)
        print(f"Validation batch - x shape: {x_val.shape}, y shape: {y_val.shape}")
        
        # Decode a sample to verify format
        enc = tiktoken.get_encoding("gpt2")
        sample_text = enc.decode(x[0].tolist())
        print(f"\nSample decoded text:\n{sample_text}")
        
    except Exception as e:
        print(f"Error getting batch: {e}")
        print("This might happen if the dataset is too small for the block size.")
