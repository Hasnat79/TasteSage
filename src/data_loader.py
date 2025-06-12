# imports
import os
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
import tiktoken
import pandas as pd

enc = tiktoken.get_encoding("gpt2")
def process(example):
        # enc = tiktoken.get_encoding("gpt2")
        text = example['text']+" Stars: "+ str(example['stars']) # Append stars to the text
        ids = enc.encode_ordinary(text)  # encode_ordinary ignores any special tokens
        out = {'ids':ids, 'len': len(ids)}
        return out
class YelpFoodBusinessReviewsDataset:
    def __init__(self, food_reviews_dataset, data_dir):
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        self.ds = food_reviews_dataset
        print(f"loaded dataset with splits: {self.ds.keys()}")
        self.save_to_disk(data_dir)
    
    def save_to_disk(self, dir):
        train_path = f"{dir}/train.bin"
        if not os.path.exists(train_path):
            self.tokenized = self.ds.map(
            process,
            remove_columns=['text', 'business_id', 'stars'],
            desc="tokenizing the splits",
            num_proc=8,
        )
            # concatenate all the ids in each dataset into one large file we can use for training
            for split, dset in self.tokenized.items():
                arr_len = np.sum(dset['len'], dtype=np.uint64)
                print(f'Processing split: {split}, total samples: {len(dset)}, total tokens: {arr_len}')
                filename = f'{dir}/{split}.bin'
                if not os.path.exists(dir):
                    os.makedirs(dir)
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

if __name__ == "__main__":

    yelp_food_business_review_dataset_hf = load_dataset("hasnat79/yelp_food_business_review_dataset_2M")

    yelp_food_review_dataset = YelpFoodBusinessReviewsDataset(yelp_food_business_review_dataset_hf, data_dir="../data/yelp_food_business_review_dataset")

