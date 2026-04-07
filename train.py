import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.optim import AdamW
from src.dataset import HonestVLMDataset, prepare_coco_bbox_data
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=str, choices=["phase1", "phase2"], default="phase1")
    parser.add_argument("--num_samples", type=int, default=2000)
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
        default="validation",
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
    args = parser.parse_args()

    # 1. Initialize multi-node environment with Gradient Accumulation
    accelerator = Accelerator(gradient_accumulation_steps=4)

    if args.phase == "phase2":
        if args.init_model_path is None:
            args.init_model_path = f"{args.output_dir.rstrip('/')}/phase1-recognition"
    load_path = args.init_model_path if args.init_model_path else args.model_id

    # 3. Setup Dataset and Dataloader
    with accelerator.main_process_first():
        real_data = prepare_coco_bbox_data(
            split=args.coco_split,
            num_samples=args.num_samples,
            phase=args.phase,
            corruption_prob=args.corruption_prob,
            blur_radius=args.blur_radius,
            min_bbox_area=args.min_bbox_area,
            seed=args.seed,
            verbose=accelerator.is_main_process
        )

    clean_count = sum(1 for item in real_data if not item.get("is_corrupted", False))
    corrupt_count = len(real_data) - clean_count
    if accelerator.is_main_process:
        print(
            f"Prepared {len(real_data)} training samples for {args.phase} "
            f"(clean={clean_count}, corrupted={corrupt_count})"
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

    dataset = HonestVLMDataset(real_data, processor)

    # 2 images per GPU * 4 GPUs = 8 images per step.
    # 8 images * 4 accumulation steps = 32 Effective Global Batch Size.
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr) # Smaller LR for fine-tuning

    # 5. Distribute everything across the 2 Nodes (4 GPUs)
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    # 6. The Multi-Node Training Loop
    epochs = args.epochs
    model.train()
    
    if accelerator.is_main_process:
        print(f"Starting training across {accelerator.num_processes} GPUs!")

    for epoch in range(epochs):
        if accelerator.is_main_process:
            print(
                f"Epoch {epoch} | dataset mix: clean={clean_count}, corrupted={corrupt_count}"
            )
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                pixel_values = batch["pixel_values"].to(dtype=torch.bfloat16)
                outputs = model(
                    input_ids=batch["input_ids"],
                    pixel_values=pixel_values,
                    labels=batch["labels"]
                )
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                
            # Log only on Node 0, GPU 0
            if accelerator.sync_gradients and accelerator.is_main_process:
                if step % 10 == 0:
                    print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

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