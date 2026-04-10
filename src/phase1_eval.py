"""
Phase 1 evaluation: bbox-conditioned COCO category prediction on a held-out split.

Metrics: overall accuracy, macro / micro F1, per-class precision/recall/F1,
confusion matrix (including unknown predictions), accuracy by bbox area bucket,
optional comparison to a baseline checkpoint.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from src.dataset import COCO_CATEGORIES, prepare_coco_bbox_data


def normalize_label(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().lower().split())


def _class_name_to_idx() -> Dict[str, int]:
    return {name.strip().lower(): i for i, name in enumerate(COCO_CATEGORIES)}


def pred_to_class_index(pred_norm: str) -> int:
    """Map normalized prediction to 0..79, or 80 if no exact COCO name match."""
    m = _class_name_to_idx()
    return m.get(pred_norm, 80)


def extract_predicted_class(pred_raw: str) -> str:
    """
    Map free-form model output to a COCO class label when possible.
    Falls back to empty string if no class name can be recovered.
    """
    pred_norm = normalize_label(pred_raw)
    if not pred_norm:
        return ""

    name_to_idx = _class_name_to_idx()
    if pred_norm in name_to_idx:
        return pred_norm

    # Strip simple punctuation around words for cases like "cat." or "a cat".
    cleaned = pred_norm.replace(",", " ").replace(".", " ").replace(":", " ").replace(";", " ")
    cleaned = " ".join(cleaned.split())
    if cleaned in name_to_idx:
        return cleaned

    # Prefer longest names first ("traffic light" before "light", etc.).
    categories_sorted = sorted(COCO_CATEGORIES, key=len, reverse=True)
    padded = f" {cleaned} "
    for name in categories_sorted:
        token = f" {name} "
        if token in padded:
            return name
    return ""


def _generate_answer(
    model,
    processor,
    prompt: str,
    image,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device=device, dtype=dtype) if k == "pixel_values" else v.to(device) for k, v in inputs.items()}
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


def _bbox_area(sample: dict) -> float:
    x1, y1, x2, y2 = sample["bbox"]
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _area_bucket(area: float) -> str:
    # COCO-style area splits (pixels^2): small < 32^2, medium < 96^2, else large.
    if area < 32 * 32:
        return "small_lt_32^2"
    if area < 96 * 96:
        return "medium_32^2_to_96^2"
    return "large_gte_96^2"


def _build_confusion_and_lists(
    samples: List[dict], predictions: List[str]
) -> Tuple[np.ndarray, List[int], List[int]]:
    """81x81 confusion: rows=true 0..79, cols=pred 0..79 or 80=unknown."""
    n_classes = len(COCO_CATEGORIES) + 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    y_true: List[int] = []
    y_pred: List[int] = []
    name_to_idx = _class_name_to_idx()
    for s, pred_raw in zip(samples, predictions):
        gt = normalize_label(s["label"])
        t_idx = name_to_idx.get(gt)
        if t_idx is None:
            continue
        p_label = extract_predicted_class(pred_raw)
        p_idx = pred_to_class_index(p_label)
        cm[t_idx, p_idx] += 1
        y_true.append(t_idx)
        y_pred.append(p_idx)
    return cm, y_true, y_pred


def _macro_micro_f1_from_cm(cm: np.ndarray) -> Tuple[float, float, List[dict]]:
    """Multiclass F1 from confusion restricted to true classes 0..79 (81st column = unknown preds)."""
    c = 80
    per_class = []
    f1s = []
    precisions = []
    recalls = []
    for i in range(c):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - tp)
        fn = float(cm[i, :].sum() - tp)
        support = float(cm[i, :].sum())
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class.append(
            {
                "class": COCO_CATEGORIES[i],
                "support": int(support),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            }
        )
        if support > 0:
            f1s.append(f1)
            precisions.append(p)
            recalls.append(r)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    total = cm[:c, :].sum()
    correct = float(np.trace(cm[:c, :c]))
    micro_p = micro_r = micro_f1 = correct / total if total > 0 else 0.0

    return macro_f1, micro_f1, per_class


def run_phase1_evaluation(
    checkpoint_path: str,
    eval_split: str = "validation",
    eval_num_samples: int = 500,
    eval_seed: int = 42,
    min_bbox_area: float = 32.0,
    baseline_model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a Phase-1 checkpoint on COCO bbox→label samples (no blur).

    If baseline_model_path is set (e.g. ``microsoft/florence-2-large``), runs the same
    metrics on the baseline for comparison.
    """
    if eval_num_samples <= 0:
        return {"skipped": True, "reason": "eval_num_samples <= 0"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if verbose:
        print(
            f"[Phase1 eval] Building {eval_num_samples} validation samples "
            f"(split={eval_split}, seed={eval_seed})..."
        )
    samples = prepare_coco_bbox_data(
        split=eval_split,
        num_samples=eval_num_samples,
        phase="phase1",
        corruption_prob=0.0,
        blur_radius=8.0,
        min_bbox_area=min_bbox_area,
        seed=eval_seed,
        verbose=verbose,
    )

    def eval_model(model_path: str, tag: str) -> Dict[str, Any]:
        if verbose:
            print(f"[Phase1 eval] Loading model: {model_path} ({tag})")
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device)
        model.eval()

        preds: List[str] = []
        for i, sample in enumerate(samples):
            pred = _generate_answer(
                model, processor, sample["prompt"], sample["image"], device, dtype
            )
            preds.append(pred)
            if verbose and (i + 1) % 50 == 0:
                print(f"[Phase1 eval] {tag}: {i + 1}/{len(samples)} predictions")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        overall_correct = sum(
            1
            for s, p in zip(samples, preds)
            if extract_predicted_class(p) == normalize_label(s["label"])
        )
        overall_acc = overall_correct / max(1, len(samples))

        cm, _, _ = _build_confusion_and_lists(samples, preds)
        macro_f1, micro_f1, per_class = _macro_micro_f1_from_cm(cm)

        bucket_stats: Dict[str, Dict[str, int]] = {}
        for s, p in zip(samples, preds):
            b = _area_bucket(_bbox_area(s))
            if b not in bucket_stats:
                bucket_stats[b] = {"total": 0, "correct": 0}
            bucket_stats[b]["total"] += 1
            if extract_predicted_class(p) == normalize_label(s["label"]):
                bucket_stats[b]["correct"] += 1
        bucket_accuracy = {
            k: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / max(1, v["total"]), 4),
            }
            for k, v in bucket_stats.items()
        }

        errors_sample: List[Dict[str, str]] = []
        for s, p in zip(samples, preds):
            pred_label = extract_predicted_class(p)
            gt_label = normalize_label(s["label"])
            if pred_label != gt_label:
                errors_sample.append(
                    {
                        "ground_truth": gt_label,
                        "prediction": pred_label,
                        "raw_prediction": normalize_label(p),
                        "prompt": s["prompt"],
                    }
                )
            if len(errors_sample) >= 20:
                break

        return {
            "tag": tag,
            "model_path": model_path,
            "num_samples": len(samples),
            "overall_accuracy": round(overall_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "micro_f1": round(micro_f1, 4),
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_shape": list(cm.shape),
            "confusion_note": "rows=true_class 0..79, cols=pred_class 0..79 or 80=unknown_pred",
            "accuracy_by_bbox_area_bucket": bucket_accuracy,
            "errors_sample": errors_sample,
            "predictions": preds,
        }

    out: Dict[str, Any] = {
        "eval_split": eval_split,
        "eval_num_samples_requested": eval_num_samples,
        "eval_seed": eval_seed,
        "min_bbox_area": min_bbox_area,
    }

    out["finetuned"] = eval_model(checkpoint_path, "finetuned")

    if baseline_model_path:
        out["baseline"] = eval_model(baseline_model_path, "baseline")
        ft = out["finetuned"]["overall_accuracy"]
        bl = out["baseline"]["overall_accuracy"]
        out["accuracy_delta_finetuned_minus_baseline"] = round(ft - bl, 4)

    # Drop heavy prediction lists from saved JSON (keep metrics only).
    def strip_preds(d: Dict[str, Any]) -> Dict[str, Any]:
        d = dict(d)
        d.pop("predictions", None)
        return d

    out_save = {
        k: strip_preds(v) if isinstance(v, dict) and "predictions" in v else v
        for k, v in out.items()
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "phase1_eval_metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out_save, f, indent=2, ensure_ascii=True)
        if verbose:
            print(f"[Phase1 eval] Wrote metrics to {path}")

    if verbose:
        print("[Phase1 eval] Summary (finetuned):")
        print(
            json.dumps(
                {
                    "overall_accuracy": out["finetuned"]["overall_accuracy"],
                    "macro_f1": out["finetuned"]["macro_f1"],
                    "micro_f1": out["finetuned"]["micro_f1"],
                    "accuracy_by_bbox_area_bucket": out["finetuned"]["accuracy_by_bbox_area_bucket"],
                },
                indent=2,
            )
        )
        if baseline_model_path and "baseline" in out:
            print("[Phase1 eval] Summary (baseline):")
            print(
                json.dumps(
                    {
                        "overall_accuracy": out["baseline"]["overall_accuracy"],
                        "macro_f1": out["baseline"]["macro_f1"],
                        "micro_f1": out["baseline"]["micro_f1"],
                    },
                    indent=2,
                )
            )
            print(
                f"[Phase1 eval] Delta accuracy (finetuned - baseline): "
                f"{out.get('accuracy_delta_finetuned_minus_baseline')}"
            )

    return out
