import argparse
import ast
import json
import os
import random

import torch
from datasets import load_dataset
from PIL import Image, ImageFilter
from transformers import AutoModelForCausalLM, AutoProcessor


def extract_question(item):
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

    return None


def extract_image(item):
    image = item.get("image")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict):
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
    return None


def blur_image(image, radius=8.0):
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))


def generate_answer(model, processor, question, image):
    inputs = processor(text=question, images=image, return_tensors="pt").to(
        "cuda", torch.bfloat16
    )
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=80,
            do_sample=False,
        )
    return processor.batch_decode(generated_ids, skip_special_tokens=False)[0]


def run_evaluation(num_samples=20, blur_radius=8.0):
    print("Loading baseline model...")
    baseline_model_path = "microsoft/florence-2-large"
    baseline_processor = AutoProcessor.from_pretrained(
        baseline_model_path, trust_remote_code=True
    )
    baseline_model = AutoModelForCausalLM.from_pretrained(
        baseline_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()
    baseline_model.eval()

    print("Loading fine-tuned model...")
    finetuned_model_path = "./honest-vlm-checkpoint"
    finetuned_processor = AutoProcessor.from_pretrained(
        finetuned_model_path, trust_remote_code=True
    )
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        finetuned_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()
    finetuned_model.eval()

    print("Loading RLHF-V dataset...")
    dataset = load_dataset(
        "openbmb/RLHF-V-Dataset",
        split="train",
        cache_dir="/netpool/homes/hathreya/honest-vlm/cache",
    )

    output_dir = "./eval_outputs"
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.jsonl")

    indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))

    with open(output_file, "w", encoding="utf-8") as f:
        for i, idx in enumerate(indices):
            item = dataset[idx]
            image = extract_image(item)
            question = extract_question(item)
            if image is None or question is None:
                continue

            blurred = blur_image(image, radius=blur_radius)
            image_name = f"sample_{i:04d}.png"
            image_path = os.path.join(image_dir, image_name)
            blurred.save(image_path)

            baseline_output = generate_answer(
                baseline_model, baseline_processor, question, blurred
            )
            finetuned_output = generate_answer(
                finetuned_model, finetuned_processor, question, blurred
            )

            row = {
                "dataset_index": int(idx),
                "question": question,
                "blurred_image_path": image_path,
                "baseline_output": baseline_output,
                "finetuned_output": finetuned_output,
            }
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            print(f"[{i + 1}/{len(indices)}] compared {image_name}")

    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--blur_radius", type=float, default=8.0)
    args = parser.parse_args()
    run_evaluation(num_samples=args.num_samples, blur_radius=args.blur_radius)