"""
MisoTTS Tamil training on Lightning AI Studios.

Uses PyTorch Lightning for:
  - Multi-GPU distributed training (DDP)
  - Automatic mixed precision (BF16)
  - Gradient checkpointing
  - Auto-resume from checkpoints
  - Logging to Lightning AI dashboard

Usage:
  # On Lightning AI Studio (already has GPU):
  pip install -e ".[train]"
  python train_lightning.py

  # Or via Lightning CLI:
  lightning run gpu train_lightning.py --gpus 2
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from models import Model, MISO_TTS_8B_CONFIG
from generator import load_llama3_tokenizer, _load_model
from moshi_compat import patch_bitsandbytes_import_for_unquantized_layers
from training.train import setup_model_for_training, LoRALinear, merge_lora_weights
from training.dataset import TamilTTSDataset, TamilTTSCollator


class MisoTTSTamilModule(pl.LightningModule):
    """PyTorch Lightning module for MisoTTS Tamil fine-tuning."""

    def __init__(
        self,
        base_model: str = "MisoLabs/MisoTTS",
        strategy: str = "lora",
        lora_rank: int = 32,
        lora_alpha: float = 64.0,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
    ):
        super().__init__()
        self.save_hyperparameters()

        patch_bitsandbytes_import_for_unquantized_layers()
        self.model = _load_model(base_model, MISO_TTS_8B_CONFIG, device="cpu", dtype=torch.bfloat16)

        config = {
            "strategy": strategy,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
        }
        self.model, _ = setup_model_for_training(self.model, config)

    def forward(self, tokens, tokens_mask, targets, targets_mask, decoder_idx):
        return self.model(tokens, tokens_mask, targets, targets_mask, decoder_idx)

    def training_step(self, batch, batch_idx):
        tokens = batch["tokens"]
        tokens_mask = batch["tokens_mask"]
        targets = batch["targets"]
        targets_mask = batch["targets_mask"]
        decoder_idx = batch["decoder_idx"]

        _, _, c0_loss, c1_loss, loss = self.model(
            tokens=tokens,
            tokens_mask=tokens_mask,
            targets=targets,
            targets_mask=targets_mask,
            decoder_idx=decoder_idx,
        )

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/c0_loss", c0_loss)
        self.log("train/c1_loss", c1_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.95),
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint):
        # Only save trainable params for LoRA
        if self.hparams.strategy == "lora":
            trainable = {}
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    trainable[name] = param.data
            checkpoint["trainable_state"] = trainable


class TamilTTSDataModule(pl.LightningDataModule):
    """Lightning DataModule for Tamil TTS data."""

    def __init__(
        self,
        manifest_path: str = "processed/manifest.jsonl",
        batch_size: int = 1,
        num_workers: int = 4,
        device: str = "cuda",
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device_str = device
        self.text_tokenizer = None
        self.audio_tokenizer = None

    def setup(self, stage=None):
        self.text_tokenizer = load_llama3_tokenizer()

        from moshi.models import loaders
        from huggingface_hub import hf_hub_download
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=self.device_str)
        mimi.set_num_codebooks(MISO_TTS_8B_CONFIG.audio_num_codebooks)
        mimi.eval()
        self.audio_tokenizer = mimi

        self.dataset = TamilTTSDataset(
            manifest_path=self.manifest_path,
            text_tokenizer=self.text_tokenizer,
            audio_tokenizer=self.audio_tokenizer,
            num_codebooks=MISO_TTS_8B_CONFIG.audio_num_codebooks,
        )

    def train_dataloader(self):
        collator = TamilTTSCollator(
            text_tokenizer=self.text_tokenizer,
            audio_tokenizer=self.audio_tokenizer,
            num_codebooks=MISO_TTS_8B_CONFIG.audio_num_codebooks,
            device=self.device_str,
        )
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collator,
            drop_last=True,
        )


def main():
    # Prepare data if not already done
    manifest = Path("processed/manifest.jsonl")
    if not manifest.exists():
        print("Preparing data first...")
        os.system("python prepare_data.py --data-dir data/ --output-dir processed/")

    if not manifest.exists():
        print("ERROR: No manifest found. Run download_data.py and prepare_data.py first.")
        sys.exit(1)

    # Lightning module
    model = MisoTTSTamilModule(
        base_model="MisoLabs/MisoTTS",
        strategy="lora",
        lora_rank=32,
        lora_alpha=64.0,
        learning_rate=2e-4,
    )

    # Data
    data = TamilTTSDataModule(
        manifest_path=str(manifest),
        batch_size=1,
        num_workers=4,
    )

    # Callbacks
    checkpoint_cb = ModelCheckpoint(
        dirpath="outputs/lightning-tamil",
        filename="misotts-tamil-{epoch:02d}-{train/loss:.4f}",
        save_top_k=3,
        monitor="train/loss",
        mode="min",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        precision="bf16-mixed",
        accumulate_grad_batches=8,
        gradient_clip_val=1.0,
        callbacks=[checkpoint_cb, lr_monitor],
        log_every_n_steps=10,
        default_root_dir="outputs/lightning-tamil",
    )

    # Train
    trainer.fit(model, data)

    # Merge and save
    print("\nMerging LoRA weights...")
    merge_lora_weights(model.model)
    output_dir = Path("outputs/lightning-tamil/final")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
        save_file(model.model.state_dict(), str(output_dir / "model_merged.safetensors"))
    except ImportError:
        torch.save(model.model.state_dict(), str(output_dir / "model_merged.pt"))
    print(f"Final model saved to {output_dir}")


if __name__ == "__main__":
    main()
