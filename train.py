import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from torch.utils.data import DataLoader
from accelerate import Accelerator
from torch.optim import AdamW
from src.dataset import HonestVLMDataset, prepare_honest_vlm_data
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--corruption_prob", type=float, default=0.3)
    args = parser.parse_args()

    # 1. Initialize multi-node environment with Gradient Accumulation
    accelerator = Accelerator(gradient_accumulation_steps=4)
    
    if accelerator.is_main_process:
        print("Loading Florence-2-large...")

    # 2. Load Model & Processor
    model_id = "microsoft/florence-2-large"
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 # A4000s support bfloat16, saving massive VRAM
    )

    # 3. Setup Dataset and Dataloader
    with accelerator.main_process_first():
        real_data = prepare_honest_vlm_data(
            num_samples=args.num_samples,
            corruption_prob=args.corruption_prob,
            verbose=accelerator.is_main_process
        )
    
    dataset = HonestVLMDataset(real_data, processor)
    
    # 2 images per GPU * 4 GPUs = 8 images per step.
    # 8 images * 4 accumulation steps = 32 Effective Global Batch Size.
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # 4. Optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5) # Smaller LR for fine-tuning

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
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
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
    
    if accelerator.is_main_process:
        unwrapped_model.save_pretrained("./honest-vlm-checkpoint")
        processor.save_pretrained("./honest-vlm-checkpoint")
        print("Training complete! Model saved on Master Node.")

if __name__ == "__main__":
    main()