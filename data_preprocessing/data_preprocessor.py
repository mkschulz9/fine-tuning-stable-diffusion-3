import random, requests, os
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from datasets import load_dataset, Dataset

class DataPreprocessor:
    """Base class for processing HuggingFace datasets"""
    def __init__(self, dataset_id: str):
        """Load dataset from HuggingFace"""
        self.dataset = load_dataset(dataset_id)
        
        print(f"\nDataset loaded: {dataset_id}")

    def reduce_caption_to_first_sentence(self):
        """Reduce caption to first sentence for given image."""
        reduced_captions = self.dataset.map(lambda sample: sample['caption'].split('.')[0] + '.')
        self.dataset['caption'] = reduced_captions

    def analyze_data(self):
        """Analyze and print information about dataset"""
        print("\n***ANALYZING DATASET***")
        print(f"Dataset Structure: {self.dataset}")
            
        print("\nChecking for missing values:")
        for feature in self.dataset['train'].features:
            num_missing = sum(1 for x in self.dataset['train'][feature] if x is None)
            print(f"{feature}: {num_missing} missing values")
        
    def visualize_data(self):
        """Visualize data by showing a random example"""
        print("\n***VISUALIZING DATASET***")
        random_index = random.sample(range(len(self.dataset["train"])), 1)
        print("Random sample from dataset:")
        for key, value in self.dataset["train"][random_index[0]].items():
            print(f"{key}: {value}")
        
        image_url = self.dataset["train"][random_index[0]]["image_url"]   
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def first_sentence(self, text):
        """Extract the first sentence from text."""
        sentence = text.split('.')[0]
        return sentence if sentence else text

    def reduce_captions(self):
        """Reduce all captions to their first sentence."""

        print("\n***REDUCING CAPTIONS TO FIRST SENTENCE***")
        self.dataset = self.dataset.map(lambda example: {'caption': self.first_sentence(example['caption'])+'.'})

    def remove_columns_split_dataset(self, train_size: float = 0.8, column_names: list[str] = None):
        """Remove columns & split dataset into train and test sets"""
        print("\n***REMOVING COLUMNS & SPLITTING DATASET***")
        if column_names != None: self.dataset = self.dataset.remove_columns(column_names)
        shuffled_dataset = self.dataset["train"].shuffle()
        dataset_split = shuffled_dataset.train_test_split(train_size=train_size)
        print(f"Columns removed and dataset successfully split: {dataset_split}")

        return dataset_split

    def create_textual_inversion_dataset(self, train_dataset, size=5):
        print("\n***CREATING TEXTUAL INVERSION DATASET***")
        save_dir = f"../datasets/{size}_imgs"
        
        os.makedirs(save_dir, exist_ok=True)

        for idx, sample in enumerate(train_dataset):
            if idx >= size:
                break
            image_path = os.path.join(save_dir, f"{sample['id']}.jpg")
            # caption_path = os.path.join(save_dir, f"{sample['id']}.txt")

            response = requests.get(sample["image_url"])
            image = Image.open(BytesIO(response.content))

            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(image_path)

            # with open(caption_path, 'w') as f:
            #     f.write(sample["caption"])
        print("***DATASET SAVED TO DATASETS DIRECTORY***")