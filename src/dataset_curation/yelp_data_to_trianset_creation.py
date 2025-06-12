import json 
import tiktoken 
import os
from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from huggingface_hub import login
from datasets import load_from_disk
# device = "cuda" if torch.cuda.is_available() else "cpu"
# device_type = 'cuda' if 'cuda' in device else 'cpu'


class BusinessDescriptionYelpDataset:
    def __init__(self):
        self.BUSINESS_DESCRIPTION_FILE = "../../data/raw/Yelp_JSON/yelp_academic_dataset_business.json"
        self.business_desc = self.load_business_description()
        self.fig_base_dir = "../../figures/data/"

    def load_business_description(self):
        """
        Load business descriptions from the JSONL file
        
        Returns:
            list: List of business description dictionaries
        """
        business_data = []
        with open(self.BUSINESS_DESCRIPTION_FILE, 'r') as f:
            for line in f:
                data = json.loads(line)
                business_data.append(data)
        print(f"Loaded {len(business_data)} businesses")
        return business_data
        
    def __getitem__(self, idx):
        """
        Get a business description by index
        
        Args:
            idx (int): Index of the business
            
        Returns:
            dict: Business description dictionary
        """
        return self.business_desc[idx]
        
    def __len__(self):
        """
        Get the number of business descriptions
        
        Returns:
            int: Number of business descriptions
        """
        return len(self.business_desc)
    def find_food_business_keywords(self):
        """
        Find food business keywords from the business descriptions
        
        Returns:
            list: List of food business keywords
        """
        food_business_keywords = []
        for data_dict in self.business_desc:
            if data_dict['categories'] is None:
                continue
            if "Food" not in data_dict['categories'] and "Restaurants" not in data_dict['categories']:
                continue
            food_business_keywords.append(data_dict['categories'])
        return food_business_keywords
    def find_unique_categories(self,categories_list):
        """
        Find unique categories and their counts from a list of categories.
        
        Args:
            categories_list (list): List of categories.
            
        Returns:
            dict: Dictionary with unique categories as keys and their counts as values.
        """
        unique_categories = {}
        for categories in categories_list:
            if categories is None:
                continue
            for category in categories.split(', '):
                if category not in unique_categories:
                    unique_categories[category] = 0
                unique_categories[category] += 1
        return unique_categories
    def plot_categories_counts(self, categories_counts, fig_name):
        """
        Plot the counts of unique categories.
        
        Args:
            categories_counts (list): List of tuples with category and its count.
        """
        
        
        categories, counts = zip(*categories_counts)
        plt.figure(figsize=(16, 12))
        plt.barh(categories, counts, color='skyblue')
        plt.xlabel('Count')
        plt.title('Counts of Unique Categories')
        plt.show()
        plt.savefig(self.fig_base_dir+fig_name)

class FoodBusinessData(BusinessDescriptionYelpDataset):
    def __init__(self):
        super().__init__()
        self.food_business_desc = self.load_food_business_description()
    def load_food_business_description(self):
        business_data = []
        for data in self.business_desc:
            if data['categories'] is None:
                continue
            if "Food" not in data['categories'] and "Restaurants" not in data['categories']:
                continue
            business_data.append(data)
        print(f"Loaded {len(business_data)} food businesses")
        return business_data
    def __getitem__(self, idx):
        """
        Get a food business description by index
        
        Args:
            idx (int): Index of the food business
            
        Returns:
            dict: Food business description dictionary
        """
        return self.food_business_desc[idx]

class ReviewsYelpDataset():
    def __init__(self):
        self.REVIEWS_FILE = "../../data/raw/Yelp_JSON/yelp_academic_dataset_review.json"
        self.reviews = self.load_reviews()
        
    def load_reviews(self):
        """
        Load reviews from the JSONL file
        
        Returns:
            list: List of review dictionaries
        """
        reviews_data = []
        with open(self.REVIEWS_FILE, 'r') as f:
            for line in f:
                data = json.loads(line)
                reviews_data.append(data)
        print(f"Loaded {len(reviews_data)} reviews")
        return reviews_data
    def __len__(self):
        return len(self.reviews)
        
    def __getitem__(self, idx):
        return self.reviews[idx]

class ReviewsToDataset:
    def __init__(self, reviews_dataset, food_business_data):
        self.reviews_dataset = reviews_dataset
        self.food_business_data = food_business_data
        self.business_ids = set(item['business_id'] for item in food_business_data.food_business_desc)
        
    def filter_food_reviews(self):
        """
        Filter reviews to only include those for food businesses
        
        Returns:
            list: List of reviews for food businesses
        """
        food_reviews = []
        for review in self.reviews_dataset.reviews:
            if review['business_id'] in self.business_ids:
                food_reviews.append(review)
        print(f"Filtered {len(food_reviews)} reviews for food businesses")
        return food_reviews
        
    def create_hf_dataset(self, max_reviews=None, train_ratio=0.8):
        """
        Convert reviews to a Hugging Face dataset, split into train and validation sets
        
        Args:
            max_reviews (int, optional): Maximum number of reviews to include. Defaults to None (all reviews).
            train_ratio (float, optional): Ratio of data to use for training. Defaults to 0.8.
            
        Returns:
            dict: Dictionary containing train and validation datasets
        """
        food_reviews = self.filter_food_reviews()
        
        if max_reviews is not None:
            food_reviews = food_reviews[:max_reviews]
            
        # Create dataset dictionary
        dataset_dict = {
            'business_id': [],
            'stars': [],
            'text': [],
        }
        
        for review in food_reviews:
            dataset_dict['business_id'].append(review['business_id'])
            dataset_dict['stars'].append(review['stars'])
            dataset_dict['text'].append(review['text'])
            
        # Convert to HF dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Split into train and validation sets
        dataset = dataset.shuffle(seed=42)
        split_dataset = dataset.train_test_split(test_size=1.0-train_ratio, seed=42)
        print(f"Train set: {len(split_dataset['train'])} examples")
        print(f"Validation set: {len(split_dataset['test'])} examples")
        
        return split_dataset
        
    def save_dataset(self, dataset, output_dir="/scratch/user/hasnat.md.abdullah/TasteSage/data/yelp_food_business_review_dataset/hf_dataset"):
        """
        Save the dataset to disk
        
        Args:
            dataset: Hugging Face dataset or dataset dictionary
            output_dir (str, optional): Output directory. Defaults to processed data directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if isinstance(dataset, dict) and 'train' in dataset and 'test' in dataset:
            dataset['train'].save_to_disk(os.path.join(output_dir, "food_reviews_dataset_train"))
            dataset['test'].save_to_disk(os.path.join(output_dir, "food_reviews_dataset_val"))
            print(f"Train dataset saved to {os.path.join(output_dir, 'food_reviews_dataset_train')}")
            print(f"Validation dataset saved to {os.path.join(output_dir, 'food_reviews_dataset_val')}")
        else:
            dataset.save_to_disk(os.path.join(output_dir, "food_reviews_dataset"))
            print(f"Dataset saved to {os.path.join(output_dir, 'food_reviews_dataset')}")



if __name__ == "__main__":
    food_business_data = FoodBusinessData()
    reviews_yelp_dataset = ReviewsYelpDataset()
    reviews_to_dataset = ReviewsToDataset(reviews_yelp_dataset, food_business_data)
    food_reviews_dataset = reviews_to_dataset.create_hf_dataset(max_reviews=2007040)# 2007040
    path = "../../data/yelp_food_business_review_dataset/hf_dataset"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        food_reviews_dataset.save_to_disk(path)

    # Upload the dataset to Hugging Face Hub

    # Load the saved dataset
    loaded_dataset = load_from_disk(path)

    # Function to upload to HF Hub
    # uncomment with your hugging face repo id to upload your custom size dataset
    # def upload_to_huggingface(dataset, repo_id):
    #     print(f"Uploading dataset to Hugging Face Hub: {repo_id}")
        
    #     # You might need to login first (uncomment if needed)
    #     # login()  # This will prompt for token if not already logged in
        
    #     dataset.push_to_hub(repo_id, private=False)
    #     print(f"Successfully uploaded dataset to {repo_id}")

    # Upload the dataset to the specified repository
    # upload_to_huggingface(loaded_dataset, "hasnat79/yelp_food_business_review_dataset_2M")

    # load and print some examples
    print(f"food_reviews_dataset: {len(food_reviews_dataset)} reviews")
    print(f"food_review_dataset: {food_reviews_dataset}")



# -----------
# verbose
# Loaded 150346 businesses
# Loaded 64616 food businesses
# Loaded 6990280 reviews
# Filtered 5126140 reviews for food businesses
# Train set: 1605632 examples
# Validation set: 401408 examples
# Saving the dataset (2/2 shards): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1605632/1605632 [00:17<00:00, 93234.44 examples/s]
# Saving the dataset (1/1 shards): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 401408/401408 [00:04<00:00, 96626.68 examples/s]
# food_review_dataset: DatasetDict({
#     train: Dataset({
#         features: ['business_id', 'stars', 'text'],
#         num_rows: 1605632
#     })
#     test: Dataset({
#         features: ['business_id', 'stars', 'text'],
#         num_rows: 401408
#     })
# })
# loaded dataset with splits: dict_keys(['train', 'test'])
# Example 0: They have the best vegan food I have ever seen for...
# Example 0: My husband and I went here about a month ago and I...
# tokenizing the splits (num_proc=8): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1605632/1605632 [01:22<00:00, 19523.72 examples/s]
# tokenizing the splits (num_proc=8): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 401408/401408 [00:49<00:00, 8138.50 examples/s]
# Processing split: train, total samples: 1605632, total tokens: 211132505
# writing /scratch/user/hasnat.md.abdullah/TasteSage/data/yelp_food_business_review_dataset/train.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [00:10<00:00, 95.02it/s]
# Processing split: test, total samples: 401408, total tokens: 52720151
# writing /scratch/user/hasnat.md.abdullah/TasteSage/data/yelp_food_business_review_dataset/test.bin: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1024/1024 [00:03<00:00, 264.77it/s]





