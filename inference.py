import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from src.dataset import prepare_coco_bbox_data


def generate_answer(model, processor, prompt, image):
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        "cuda", torch.bfloat16
    )
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=80,
            do_sample=False,
        )
    new_tokens = generated_ids[:, input_len:]
    return processor.batch_decode(new_tokens, skip_special_tokens=True)[0]


def _normalize_output(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def run_evaluation(
    model_path="./checkpoints/phase2-honest",
    baseline_model_path=None,
    num_samples=200,
    coco_split="validation",
    blur_radius=8.0,
):
    print(f"Loading fine-tuned model from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda()
    model.eval()

    baseline_model = None
    baseline_processor = None
    if baseline_model_path:
        print(f"Loading baseline model from {baseline_model_path}...")
        baseline_processor = AutoProcessor.from_pretrained(
            baseline_model_path, trust_remote_code=True
        )
        baseline_model = AutoModelForCausalLM.from_pretrained(
            baseline_model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda()
        baseline_model.eval()

    output_dir = "./eval_outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.jsonl")

    clean_samples = prepare_coco_bbox_data(
        split=coco_split,
        num_samples=num_samples,
        phase="phase1",
        blur_radius=blur_radius,
        verbose=True,
    )
    blurred_samples = prepare_coco_bbox_data(
        split=coco_split,
        num_samples=num_samples,
        phase="phase2",
        corruption_prob=1.0,
        blur_radius=blur_radius,
        verbose=True,
    )

    clean_correct = 0
    blurred_refusal = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for i, sample in enumerate(clean_samples):
            pred = _normalize_output(
                generate_answer(model, processor, sample["prompt"], sample["image"])
            )
            gt = _normalize_output(sample["label"])
            if pred == gt:
                clean_correct += 1
            row = {
                "subset": "clean",
                "sample_index": i,
                "prompt": sample["prompt"],
                "ground_truth": gt,
                "prediction": pred,
                "is_correct": pred == gt,
            }
            if baseline_model is not None:
                baseline_pred = _normalize_output(
                    generate_answer(
                        baseline_model, baseline_processor, sample["prompt"], sample["image"]
                    )
                )
                row["baseline_prediction"] = baseline_pred
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

        for i, sample in enumerate(blurred_samples):
            pred = _normalize_output(
                generate_answer(model, processor, sample["prompt"], sample["image"])
            )
            refused = pred == "unrecognizable"
            if refused:
                blurred_refusal += 1
            row = {
                "subset": "blurred",
                "sample_index": i,
                "prompt": sample["prompt"],
                "ground_truth": "unrecognizable",
                "prediction": pred,
                "is_correct": refused,
            }
            if baseline_model is not None:
                baseline_pred = _normalize_output(
                    generate_answer(
                        baseline_model, baseline_processor, sample["prompt"], sample["image"]
                    )
                )
                row["baseline_prediction"] = baseline_pred
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    clean_acc = clean_correct / max(1, len(clean_samples))
    blurred_refusal_rate = blurred_refusal / max(1, len(blurred_samples))
    metrics = {
        "clean_samples": len(clean_samples),
        "blurred_samples": len(blurred_samples),
        "clean_label_accuracy": clean_acc,
        "blurred_unrecognizable_rate": blurred_refusal_rate,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)

    print(json.dumps(metrics, ensure_ascii=True, indent=2))
    print(f"Saved detailed outputs to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints/phase2-honest")
    parser.add_argument("--baseline_model_path", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--coco_split", type=str, default="validation")
    parser.add_argument("--blur_radius", type=float, default=8.0)
    args = parser.parse_args()
    run_evaluation(
        model_path=args.model_path,
        baseline_model_path=args.baseline_model_path,
        num_samples=args.num_samples,
        coco_split=args.coco_split,
        blur_radius=args.blur_radius,
    )