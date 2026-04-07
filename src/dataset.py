from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
from PIL import Image
import json
import ast
import io
import random

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
    if isinstance(text_field, str):
        parsed = None
        try:
            parsed = json.loads(text_field)
        except Exception:
            try:
                parsed = ast.literal_eval(text_field)
            except Exception:
                parsed = None
        if parsed is not None:
            text_field = parsed

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


def _extract_chosen_answer(item):
    """Best-effort extraction of the preferred clean answer."""
    text_field = item.get("text")
    if isinstance(text_field, str):
        parsed = None
        try:
            parsed = json.loads(text_field)
        except Exception:
            try:
                parsed = ast.literal_eval(text_field)
            except Exception:
                parsed = None
        if parsed is not None:
            text_field = parsed

    if isinstance(text_field, dict):
        chosen = text_field.get("chosen")
        if isinstance(chosen, str) and chosen.strip():
            return chosen.strip()
    if isinstance(text_field, list):
        for entry in text_field:
            if not isinstance(entry, dict):
                continue
            chosen = entry.get("chosen")
            if isinstance(chosen, str) and chosen.strip():
                return chosen.strip()

    if "chosen" in item and isinstance(item["chosen"], str) and item["chosen"].strip():
        return item["chosen"].strip()
    return None


def _extract_image(item):
    """Get PIL image from either decoded image or HF image dict."""
    image = item.get("image")
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
    return None


def _add_gaussian_noise(image, noise_std=55.0):
    """Apply strong Gaussian noise so the full image becomes unclear."""
    image_rgb = image.convert("RGB")
    arr = np.array(image_rgb, dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=arr.shape).astype(np.float32)
    noised = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noised)


def prepare_honest_vlm_data(num_samples=2000, corruption_prob=0.3, verbose=True):
    if verbose:
        print("Loading RLHF-V dataset...")
    dataset = load_dataset(
        "openbmb/RLHF-V-Dataset",
        split="train",
        cache_dir="/netpool/homes/hathreya/honest-vlm/cache"
    )
    
    # Shuffle so we get a random mix of samples
    take_n = min(num_samples, len(dataset))
    dataset = dataset.shuffle(seed=42).select(range(take_n))
    
    processed_data = []
    corruption_prob = max(0.0, min(1.0, corruption_prob))
    corrupted_count = 0
    
    if verbose:
        print(f"Applying synthetic corruption to up to {take_n} RLHF-V samples...")
    for item in dataset:
        image = _extract_image(item)
        question = _extract_question(item)
        clean_answer = _extract_chosen_answer(item)

        # Skip samples with missing image/question/clean answer.
        if image is None or question is None or clean_answer is None:
            continue

        should_corrupt = random.random() < corruption_prob
        if should_corrupt:
            final_image = _add_gaussian_noise(image)
            final_answer = "Unknown because of clarity issues"
            corrupted_count += 1
        else:
            final_image = image
            final_answer = clean_answer

        # Use RLHF-V question as prompt with mixed clean/corrupted supervision.
        processed_data.append({
            "image": final_image,
            "prompt": question,
            "text": final_answer
        })

    if len(processed_data) == 0:
        raise ValueError(
            "No valid RLHF-V samples found after parsing. "
            "Expected image and question/chosen answer in item['text']."
        )

    if verbose:
        print(
            f"Prepared {len(processed_data)} samples "
            f"({corrupted_count} corrupted, {len(processed_data) - corrupted_count} clean)."
        )

    return processed_data
