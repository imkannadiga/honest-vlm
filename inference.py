import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

def test_model(image_path):
    print("Loading Honest-VLM...")
    model_path = "./honest-vlm-checkpoint"
    
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).cuda()

    image = Image.open(image_path).convert("RGB")
    prompt = "<MORE_DETAILED_CAPTION>"

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
    # Replace with a path to a highly pixelated or blurry image
    test_model("test_blurry_image.jpg")