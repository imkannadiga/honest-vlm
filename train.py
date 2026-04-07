import os
import pickle
import random
import argparse

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

from src.dataset import HonestVLMDataset, prepare_coco_bbox_data


def _coco_prepared_cache_path(cache_dir: str, args, num_samples) -> str:
    """Stable filename so different prep settings do not collide."""
    ns = "full" if num_samples is None else str(int(num_samples))
    name = (
        f"coco_split-{args.coco_split}_ns-{ns}_ph-{args.phase}_"
        f"cp-{args.corruption_prob}_br-{args.blur_radius}_"
        f"mba-{args.min_bbox_area}_seed-{args.seed}.pkl"
    )
    return os.path.join(cache_dir, name)


def _load_or_prepare_coco_bbox_data(accelerator: Accelerator, args, num_samples):
    """
    Single-writer preparation on shared storage: main process builds or refreshes the cache;
    all ranks load the same pickled list so COCO prep is not repeated on every process.
    """
    os.makedirs(args.prepared_data_cache_dir, exist_ok=True)
    cache_path = _coco_prepared_cache_path(args.prepared_data_cache_dir, args, num_samples)
    tmp_path = cache_path + ".tmp"

    if accelerator.is_main_process:
        if os.path.isfile(cache_path) and not args.prepared_data_force_refresh:
            print(f"Loading prepared COCO data from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                full_data = pickle.load(f)
        else:
            if args.prepared_data_force_refresh and os.path.isfile(cache_path):
                print(
                    f"Refreshing prepared COCO cache (--prepared_data_force_refresh): "
                    f"{cache_path}"
                )
            full_data = prepare_coco_bbox_data(
                split=args.coco_split,
                num_samples=num_samples,
                phase=args.phase,
                corruption_prob=args.corruption_prob,
                blur_radius=args.blur_radius,
                min_bbox_area=args.min_bbox_area,
                seed=args.seed,
                verbose=True,
            )
            with open(tmp_path, "wb") as f:
                pickle.dump(full_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)
            print(f"Saved prepared COCO data to cache: {cache_path}")

    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        with open(cache_path, "rb") as f:
            full_data = pickle.load(f)

    return full_data


def _distributed_mean_scalar(loss: torch.Tensor, accelerator: Accelerator) -> float:
    """Mean of scalar loss across processes (equal per-device batch sizes)."""
    x = loss.detach().float()
    if x.dim() == 0:
        x = x.unsqueeze(0)
    gathered = accelerator.gather_for_metrics(x)
    return float(gathered.mean().item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, choices=["phase1", "phase2"], default="phase1")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Max bbox instances from COCO split; 0 = use the full split.",
    )
    parser.add_argument(
        "--train_split_ratio",
        type=float,
        default=0.8,
        help="Fraction of prepared samples used for training (rest is held-out test).",
    )
    parser.add_argument("--coco_split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--corruption_prob", type=float, default=0.3)
    parser.add_argument("--blur_radius", type=float, default=8.0)
    parser.add_argument("--min_bbox_area", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--model_id", type=str, default="microsoft/florence-2-large")
    parser.add_argument("--init_model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--skip_phase1_eval",
        action="store_true",
        help="If set, skip automatic Phase-1 validation metrics after training.",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        help="COCO split for Phase-1 post-training eval (held-out from training).",
    )
    parser.add_argument(
        "--eval_num_samples",
        type=int,
        default=500,
        help="Number of bbox instances for Phase-1 eval (0 disables eval).",
    )
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument(
        "--no_eval_baseline",
        action="store_true",
        help="Skip baseline (pretrained) comparison in Phase-1 eval (faster).",
    )
    parser.add_argument(
        "--prepared_data_cache_dir",
        type=str,
        default="./data/prepared_coco_cache",
        help="Shared directory for pickled prepared COCO bbox samples (safe for multi-node).",
    )
    parser.add_argument(
        "--prepared_data_force_refresh",
        action="store_true",
        help="Rebuild prepared data and overwrite the cache file for this config.",
    )
    args = parser.parse_args()

    # 1. Initialize multi-node environment with Gradient Accumulation
    accelerator = Accelerator(gradient_accumulation_steps=4)

    if args.phase == "phase2":
        if args.init_model_path is None:
            args.init_model_path = f"{args.output_dir.rstrip('/')}/phase1-recognition"
    load_path = args.init_model_path if args.init_model_path else args.model_id

    # 3. Setup Dataset and Dataloader (full COCO split by default, then 80/20 train/test)
    num_samples = None if args.num_samples <= 0 else args.num_samples
    full_data = _load_or_prepare_coco_bbox_data(accelerator, args, num_samples)

    split_rng = random.Random(args.seed)
    shuffled = full_data.copy()
    split_rng.shuffle(shuffled)
    ratio = min(max(args.train_split_ratio, 0.0), 1.0)
    n_train = int(len(shuffled) * ratio)
    if n_train <= 0 or n_train >= len(shuffled):
        raise ValueError(
            f"Invalid train/test split: n={len(shuffled)}, train_split_ratio={ratio} "
            f"gives n_train={n_train}. Use a larger dataset or adjust --train_split_ratio."
        )
    train_data = shuffled[:n_train]
    test_data = shuffled[n_train:]

    train_clean = sum(1 for item in train_data if not item.get("is_corrupted", False))
    train_corrupt = len(train_data) - train_clean
    if accelerator.is_main_process:
        print(
            f"Train/test split: {len(train_data)} train, {len(test_data)} test "
            f"(ratio={ratio:.2f}). Train mix: clean={train_clean}, corrupted={train_corrupt}"
        )

    if args.dry_run:
        if accelerator.is_main_process:
            print("Dry run complete. Exiting before model/optimizer initialization.")
        return

    if accelerator.is_main_process:
        print(f"Loading model/processor from: {load_path}")

    # 2. Load Model & Processor
    processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 # A4000s support bfloat16, saving massive VRAM
    )

    train_dataset = HonestVLMDataset(train_data, processor)
    test_dataset = HonestVLMDataset(test_data, processor)

    # 2 images per GPU * 4 GPUs = 8 images per step.
    # 8 images * 4 accumulation steps = 32 Effective Global Batch Size.
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr) # Smaller LR for fine-tuning

    # 5. Distribute everything across the 2 Nodes (4 GPUs)
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    # 6. The Multi-Node Training Loop
    epochs = args.epochs
    model.train()

    if accelerator.is_main_process:
        print(f"Starting training across {accelerator.num_processes} GPUs!")

    for epoch in range(epochs):
        progress_bar = None
        if accelerator.is_main_process:
            progress_bar = tqdm(
                total=len(train_dataloader),
                desc=f"Epoch {epoch}",
                leave=True,
                dynamic_ncols=True,
            )
        if accelerator.is_main_process:
            print(
                f"Epoch {epoch} | train mix: clean={train_clean}, corrupted={train_corrupt}"
            )

        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                pixel_values = batch["pixel_values"].to(dtype=torch.bfloat16)
                outputs = model(
                    input_ids=batch["input_ids"],
                    pixel_values=pixel_values,
                    labels=batch["labels"],
                )
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            batch_mean = _distributed_mean_scalar(loss, accelerator)
            train_loss_sum += batch_mean
            train_batches += 1

            if accelerator.is_main_process and progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{batch_mean:.4f}"})

        if progress_bar is not None:
            progress_bar.close()

        avg_train_loss = train_loss_sum / max(train_batches, 1)

        model.eval()
        test_loss_sum = 0.0
        test_batches = 0
        with torch.no_grad():
            for batch in test_dataloader:
                pixel_values = batch["pixel_values"].to(dtype=torch.bfloat16)
                outputs = model(
                    input_ids=batch["input_ids"],
                    pixel_values=pixel_values,
                    labels=batch["labels"],
                )
                loss = outputs.loss
                batch_mean = _distributed_mean_scalar(loss, accelerator)
                test_loss_sum += batch_mean
                test_batches += 1
        avg_test_loss = test_loss_sum / max(test_batches, 1)
        model.train()

        if accelerator.is_main_process:
            print(
                f"Epoch {epoch} | avg train loss: {avg_train_loss:.4f} | "
                f"avg test loss: {avg_test_loss:.4f}"
            )

    # 7. Safe Checkpointing
    accelerator.wait_for_everyone() # Force Node 1 to wait for Node 0 to finish
    unwrapped_model = accelerator.unwrap_model(model)

    if args.save_name is None:
        args.save_name = "phase1-recognition" if args.phase == "phase1" else "phase2-honest"
    save_path = f"{args.output_dir.rstrip('/')}/{args.save_name}"

    if accelerator.is_main_process:
        unwrapped_model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        print(f"Training complete! Model saved at {save_path}.")

        if (
            args.phase == "phase1"
            and not args.skip_phase1_eval
            and args.eval_num_samples > 0
        ):
            from src.phase1_eval import run_phase1_evaluation

            eval_out_dir = os.path.join(save_path, "phase1_eval")
            run_phase1_evaluation(
                checkpoint_path=save_path,
                eval_split=args.eval_split,
                eval_num_samples=args.eval_num_samples,
                eval_seed=args.eval_seed,
                min_bbox_area=args.min_bbox_area,
                baseline_model_path=None if args.no_eval_baseline else args.model_id,
                output_dir=eval_out_dir,
                verbose=True,
            )

if __name__ == "__main__":
    main()