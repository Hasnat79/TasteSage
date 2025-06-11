from datasets import load_dataset
import tiktoken 
import os 
import numpy as np
from tqdm.auto import tqdm
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'

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

if __name__ == "__main__":
    dataset = TinyStoriesDataset()
    # Example usage
    # x, y = dataset.get_batch('train', batch_size=32, block_size=64)
    # print(f"x shape: {x.shape}, y shape: {y.shape}")
    # print(f"x: {x[0]}")
    # print(f"y: {y[0]}")
