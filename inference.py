import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import numpy as np


def add_gaussian_noise(image, noise_std=55.0):
    """Apply strong Gaussian noise so the full image becomes unclear."""
    image_rgb = image.convert("RGB")
    arr = np.array(image_rgb, dtype=np.float32)
    noise = np.random.normal(loc=0.0, scale=noise_std, size=arr.shape).astype(np.float32)
    noised = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noised)


def test_model(image_path, question):
    print("Loading Honest-VLM...")
    model_path = "./honest-vlm-checkpoint"
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).cuda()

    clean_image = Image.open(image_path).convert("RGB")
    image = add_gaussian_noise(clean_image)
    prompt = question

    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.bfloat16)

    print("Generating description...")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=50,
        do_sample=False
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    print(f"\nModel Output: {generated_text}")

if __name__ == "__main__":
    # Replace with your test image and question.
    test_model("test_image.jpg", "What is the person holding?")