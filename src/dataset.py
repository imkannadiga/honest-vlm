from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
from PIL import Image

class HonestVLMDataset(Dataset):
    def __init__(self, data_list, processor):
        """
        data_list: list of dicts [{'image': PIL.Image, 'prompt': str, 'text': str}]
        """
        self.data_list = data_list
        self.processor = processor
        # Florence-2 injects image tokens into the sequence. A tiny max_length
        # (e.g. 128) can become negative after internal accounting.
        self.input_max_length = 1024
        self.label_max_length = 128

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        image = item["image"]
        prompt = item["prompt"]
        text = item["text"]
        
        # The processor handles both image normalization and text tokenization simultaneously
        inputs = self.processor(
            text=prompt, 
            images=image, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.input_max_length,
            truncation=True
        )
        
        # Tokenize the target text (the label we want the model to learn)
        labels = self.processor.tokenizer(
            text=text, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=self.label_max_length,
            truncation=True
        ).input_ids

        # PyTorch cross-entropy ignores targets set to -100 (we don't want to calculate loss on padding tokens)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels.squeeze(0)
        }

def _extract_question(item):
    """Best-effort question extraction for RLHF-V style samples."""
    text_field = item.get("text")
    if isinstance(text_field, dict):
        question = text_field.get("question")
        if isinstance(question, str) and question.strip():
            return question.strip()
    if isinstance(text_field, list):
        for entry in text_field:
            if not isinstance(entry, dict):
                continue
            question = entry.get("question")
            if isinstance(question, str) and question.strip():
                return question.strip()

    if "question" in item and isinstance(item["question"], str):
        return item["question"]
    if "prompt" in item and isinstance(item["prompt"], str):
        return item["prompt"]
    if "query" in item and isinstance(item["query"], str):
        return item["query"]
    if "chosen" in item and isinstance(item["chosen"], str):
        return item["chosen"]
    if "conversations" in item and isinstance(item["conversations"], list):
        for turn in item["conversations"]:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", turn.get("role", ""))).lower()
            value = turn.get("value", turn.get("content"))
            if role in {"human", "user"} and isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _add_gaussian_noise(image, noise_std=55.0):
    """Apply strong Gaussian noise so the full image becomes unclear."""
    image_rgb = image.convert("RGB")
    arr = np.array(image_rgb, dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=arr.shape).astype(np.float32)
    noised = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noised)


def prepare_honest_vlm_data(num_samples=2000):
    print("Loading RLHF-V dataset...")
    dataset = load_dataset(
        "openbmb/RLHF-V-Dataset",
        split="train",
        cache_dir="/netpool/homes/hathreya/honest-vlm/cache"
    )
    
    # Shuffle so we get a random mix of samples
    dataset = dataset.shuffle(seed=42).select(range(num_samples))
    
    processed_data = []
    
    print(f"Applying synthetic corruption to up to {num_samples} RLHF-V samples...")
    for item in dataset:
        image = item.get("image")
        question = _extract_question(item)

        # Skip samples with missing image/question
        if image is None or question is None:
            continue

        # Apply full-image Gaussian noise so visual details become unclear.
        corrupted_image = _add_gaussian_noise(image)

        # Train refusal on corrupted visual evidence while keeping original question.
        processed_data.append({
            "image": corrupted_image,
            "prompt": question,
            "text": "Unknown because of clarity issues"
        })
        
    return processed_data
