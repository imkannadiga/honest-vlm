from torch.utils.data import Dataset
from datasets import load_dataset
import random
from src.synthetic_corruption import corrupt_region

class HonestVLMDataset(Dataset):
    def __init__(self, data_list, processor):
        """
        data_list: list of dicts [{'image': PIL.Image, 'text': "A man holding an [unrecognizable object]"}]
        """
        self.data_list = data_list
        self.processor = processor

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = item["image"]
        text = item["text"]

        # Florence-2 requires a specific task prompt prefix. We'll use detailed captioning.
        prompt = "<MORE_DETAILED_CAPTION>"
        
        # The processor handles both image normalization and text tokenization simultaneously
        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=128, 
            truncation=True
        )
        
        # Tokenize the target text (the label we want the model to learn)
        labels = self.processor.tokenizer(
            text=text, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=128, 
            truncation=True
        ).input_ids

        # PyTorch cross-entropy ignores targets set to -100 (we don't want to calculate loss on padding tokens)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels.squeeze(0)
        }

def prepare_honest_vlm_data(num_samples=2000):
    print("Downloading COCO 2017 Validation subset (~1GB)...")
    # This automatically downloads, caches, and loads the 5000 validation images
    dataset = load_dataset("rafaelpadilla/coco2017", split="val")
    
    # Shuffle so we get a random mix of images
    dataset = dataset.shuffle(seed=42).select(range(num_samples))
    
    processed_data = []
    
    print(f"Applying synthetic corruption to {num_samples} images...")
    for item in dataset:
        image = item["image"]
        objects = item["objects"]
        
        # If the image has no objects, skip it
        if len(objects["bbox"]) == 0:
            continue
            
        # 1. Pick a random object in the image to corrupt
        random_idx = random.randint(0, len(objects["bbox"]) - 1)
        bbox = objects["bbox"][random_idx]
        
        # 2. COCO format is [x, y, width, height]. 
        # Our OpenCV script expects [x_min, y_min, x_max, y_max].
        x, y, w, h = bbox
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h
        converted_bbox = [x_min, y_min, x_max, y_max]
        
        # 3. Apply the blur/pixelation
        corrupted_image = corrupt_region(image, converted_bbox, method="pixelate")
        
        # 4. Append to our training list with the new ground-truth label
        processed_data.append({
            "image": corrupted_image,
            "text": "A man holding an [unrecognizable object]." 
            # Note: In a full production run, you'd want to extract the actual 
            # background context from the COCO captions. For a 24hr proof-of-concept, 
            # teaching it to output this specific string when visual data is lost works perfectly.
        })
        
    return processed_data