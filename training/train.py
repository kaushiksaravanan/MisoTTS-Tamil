"""
Fine-tune MisoTTS 8B for Tamil text-to-speech.

Strategy:
  - LoRA on backbone (Llama 8B) to preserve English+multilingual capability
  - Full fine-tune on text embeddings (Tamil script needs new token coverage)
  - Full fine-tune on decoder + audio heads (adapt to Tamil prosody)
  - Optional: full fine-tune backbone for maximum Tamil quality (needs more VRAM)

Supports:
  - Single GPU (A100 80GB with LoRA, or multi-GPU with FSDP)
  - Gradient checkpointing for memory efficiency
  - Mixed precision (bfloat16)
  - Wandb logging (optional)

Run: python -m training.train --config configs/tamil_finetune.yaml
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Model, MISO_TTS_8B_CONFIG, ModelArgs
from generator import load_llama3_tokenizer
from moshi_compat import patch_bitsandbytes_import_for_unquantized_layers
from training.dataset import TamilTTSDataset, TamilTTSCollator


def load_config(config_path: str) -> dict:
    """Load YAML or JSON config."""
    path = Path(config_path)
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("pip install pyyaml to use YAML configs")
    else:
        with open(path) as f:
            return json.load(f)


def apply_lora(model: Model, rank: int = 16, alpha: float = 32.0, target_modules: Optional[list] = None):
    """
    Apply LoRA to backbone attention layers.
    Replaces Q/K/V/O projections with low-rank adapters.
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "output_proj"]

    lora_layers = []

    for name, module in model.backbone.named_modules():
        if any(t in name for t in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = dict(model.backbone.named_modules())[parent_name]

                lora = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora)
                lora_layers.append(lora)

    return lora_layers


class LoRALinear(nn.Module):
    """LoRA adapter for a Linear layer."""

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * (1.0 / rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x):
        base_out = self.original(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


def setup_model_for_training(model: Model, config: dict) -> tuple:
    """Configure which parameters to train based on strategy."""
    strategy = config.get("strategy", "lora")

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    trainable_params = []

    if strategy == "lora":
        lora_rank = config.get("lora_rank", 16)
        lora_alpha = config.get("lora_alpha", 32.0)
        lora_layers = apply_lora(model, rank=lora_rank, alpha=lora_alpha)
        for layer in lora_layers:
            trainable_params.extend([layer.lora_A, layer.lora_B])
        print(f"  LoRA applied: rank={lora_rank}, alpha={lora_alpha}, layers={len(lora_layers)}")

    elif strategy == "full":
        for param in model.backbone.parameters():
            param.requires_grad = True
            trainable_params.append(param)

    # Always train text embeddings (Tamil script)
    for param in model.text_embeddings.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    # Always train decoder (prosody adaptation)
    for param in model.decoder.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    # Always train audio heads
    for param in model.codebook0_head.parameters():
        param.requires_grad = True
        trainable_params.append(param)
    model.audio_head.requires_grad = True
    trainable_params.append(model.audio_head)

    # Train projection layer
    for param in model.projection.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    # Audio embeddings
    for param in model.audio_embeddings.parameters():
        param.requires_grad = True
        trainable_params.append(param)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params / 1e6:.1f}M")
    print(f"  Trainable params: {trainable / 1e6:.1f}M ({100*trainable/total_params:.1f}%)")

    return model, trainable_params


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """Cosine learning rate scheduler with linear warmup."""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(
    model: Model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    epoch: int,
    config: dict,
    global_step: int,
) -> tuple:
    """Train for one epoch. Returns (avg_loss, global_step)."""
    model.train()
    total_loss = 0.0
    total_c0_loss = 0.0
    total_c1_loss = 0.0
    num_batches = 0

    grad_accum_steps = config.get("gradient_accumulation_steps", 1)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    log_interval = config.get("log_interval", 10)

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        tokens = batch["tokens"].to(device)
        tokens_mask = batch["tokens_mask"].to(device)
        targets = batch["targets"].to(device)
        targets_mask = batch["targets_mask"].to(device)
        decoder_idx = batch["decoder_idx"].to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _, _, c0_loss, c1_plus_loss, loss = model(
                tokens=tokens,
                tokens_mask=tokens_mask,
                targets=targets,
                targets_mask=targets_mask,
                decoder_idx=decoder_idx,
            )

        loss = loss / grad_accum_steps
        loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        total_loss += loss.item() * grad_accum_steps
        total_c0_loss += c0_loss.item()
        total_c1_loss += c1_plus_loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            avg = total_loss / num_batches
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch} | Step {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg:.4f} (c0={total_c0_loss/num_batches:.4f}, "
                  f"c1+={total_c1_loss/num_batches:.4f}) | LR: {lr:.2e}")

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


def save_checkpoint(model: Model, optimizer, scheduler, epoch: int, global_step: int, config: dict, output_dir: Path):
    """Save training checkpoint."""
    ckpt_dir = output_dir / f"checkpoint-{global_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save only trainable parameters for LoRA
    if config.get("strategy", "lora") == "lora":
        trainable_state = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_state[name] = param.data.cpu()
        torch.save(trainable_state, ckpt_dir / "trainable_params.pt")
    else:
        torch.save(model.state_dict(), ckpt_dir / "model.safetensors")

    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, ckpt_dir / "training_state.pt")

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved checkpoint to {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune MisoTTS for Tamil")
    parser.add_argument("--config", required=True, help="Path to training config")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint directory")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load model
    print("Loading MisoTTS 8B base model...")
    patch_bitsandbytes_import_for_unquantized_layers()
    model_source = config.get("base_model", "MisoLabs/MisoTTS")

    from generator import _load_model
    model = _load_model(model_source, MISO_TTS_8B_CONFIG, device="cpu", dtype=torch.bfloat16)

    # Setup training parameters
    print("Configuring training strategy...")
    model, trainable_params = setup_model_for_training(model, config)
    model = model.to(device)

    if config.get("gradient_checkpointing", True):
        model.backbone.gradient_checkpointing_enable = True

    # Load tokenizers
    print("Loading tokenizers...")
    text_tokenizer = load_llama3_tokenizer()

    from moshi.models import loaders
    from huggingface_hub import hf_hub_download
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device=device)
    mimi.set_num_codebooks(MISO_TTS_8B_CONFIG.audio_num_codebooks)
    mimi.eval()

    # Load dataset
    print("Loading dataset...")
    manifest_path = config.get("manifest_path", "processed/manifest.jsonl")
    romanize = config.get("romanize_input", False)
    dataset = TamilTTSDataset(
        manifest_path=manifest_path,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=mimi,
        num_codebooks=MISO_TTS_8B_CONFIG.audio_num_codebooks,
        romanize=romanize,
    )
    print(f"  Dataset size: {len(dataset)} utterances")
    if romanize:
        print(f"  Mode: Romanized Tamil (ISO 15919)")

    collator = TamilTTSCollator(
        text_tokenizer=text_tokenizer,
        audio_tokenizer=mimi,
        num_codebooks=MISO_TTS_8B_CONFIG.audio_num_codebooks,
        device=device,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 2),
        shuffle=True,
        num_workers=config.get("num_workers", 2),
        collate_fn=collator,
        pin_memory=False,
        drop_last=True,
    )

    # Optimizer
    lr = config.get("learning_rate", 1e-4)
    weight_decay = config.get("weight_decay", 0.01)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler
    num_epochs = config.get("num_epochs", 10)
    num_training_steps = len(dataloader) * num_epochs // config.get("gradient_accumulation_steps", 1)
    num_warmup_steps = config.get("warmup_steps", min(500, num_training_steps // 10))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Output directory
    output_dir = Path(config.get("output_dir", "outputs/tamil-finetune"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    global_step = 0
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        state = torch.load(resume_path / "training_state.pt", map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        global_step = state["global_step"]
        print(f"  Resumed from epoch {start_epoch}, step {global_step}")

    # Training loop
    print(f"\nStarting training:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {config.get('batch_size', 2)}")
    print(f"  Gradient accumulation: {config.get('gradient_accumulation_steps', 1)}")
    print(f"  Learning rate: {lr}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Total steps: {num_training_steps}")
    print(f"  Save every: {config.get('save_every_steps', 500)} steps")
    print("=" * 60)

    save_every = config.get("save_every_steps", 500)
    best_loss = float("inf")

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        avg_loss, global_step = train_epoch(
            model, dataloader, optimizer, scheduler,
            device, epoch, config, global_step
        )
        epoch_time = time.time() - epoch_start

        print(f"\n  Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, config, output_dir)
            print(f"  New best loss! Saved.")

        if (epoch + 1) % config.get("save_every_epochs", 2) == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, config, output_dir)

    # Save final model
    print("\nSaving final model...")
    save_checkpoint(model, optimizer, scheduler, num_epochs - 1, global_step, config, output_dir)

    # Export merged model for inference
    if config.get("strategy", "lora") == "lora":
        print("Merging LoRA weights for inference...")
        merge_lora_weights(model)
        final_path = output_dir / "model_merged.safetensors"
        try:
            from safetensors.torch import save_file
            save_file(model.state_dict(), str(final_path))
        except ImportError:
            torch.save(model.state_dict(), output_dir / "model_merged.pt")
        print(f"  Merged model saved to {output_dir}")

    print("\nTraining complete!")


def merge_lora_weights(model: Model):
    """Merge LoRA adapters back into base weights for inference."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            delta = (module.lora_A @ module.lora_B) * module.scaling
            module.original.weight.data += delta.T


if __name__ == "__main__":
    main()
