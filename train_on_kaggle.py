"""
MisoTTS Tamil Fine-tuning on Kaggle.

This script is designed to run as a Kaggle notebook with GPU acceleration.
It handles: data download, preprocessing, tokenization, and LoRA training.

To use on Kaggle:
1. Create a new notebook with GPU (T4 x2 or P100)
2. Add datasets: vickythefire2000/indic-tts-tamil-female, vickythefire2000/indic-tts-tamil-male
3. Run this script

The fine-tuned model will be saved and can be downloaded or pushed to HuggingFace.
"""

import os
import sys
import json
import time
from pathlib import Path

# Install dependencies
os.system("pip install -q torch==2.4.0 torchaudio==2.4.0 torchtune==0.4.0 torchao==0.9.0 "
          "transformers==4.49.0 tokenizers==0.21.0 huggingface_hub==0.28.1 moshi==0.2.2 "
          "safetensors pyyaml")
os.system("pip install -q 'silentcipher @ git+https://github.com/SesameAILabs/silentcipher@d46d7d0893a583d8968ab3a6626e2289faec9152'")

# Clone repo
if not Path("MisoTTS-Tamil").exists():
    os.system("git clone https://github.com/kaushiksaravanan/MisoTTS-Tamil.git")
os.chdir("MisoTTS-Tamil")
sys.path.insert(0, ".")

import torch
import torchaudio

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# ============================================================================
# STEP 1: Prepare data from Kaggle datasets
# ============================================================================
print("\n" + "=" * 60)
print("STEP 1: Preparing training data")
print("=" * 60)

KAGGLE_DATA_DIRS = [
    "/kaggle/input/indic-tts-tamil-female",
    "/kaggle/input/indic-tts-tamil-male",
]

manifest_entries = []
processed_dir = Path("processed/audio")
processed_dir.mkdir(parents=True, exist_ok=True)

TARGET_SR = 24000

for data_dir in KAGGLE_DATA_DIRS:
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"  [SKIP] {data_dir} not found")
        continue

    speaker_name = "female_0" if "female" in data_dir else "male_0"
    speaker_id = 0 if "female" in data_dir else 1

    # Find audio/transcript pairs
    audio_dir = data_path / "audio_files"
    trans_dir = data_path / "trans_files"

    if not audio_dir.exists():
        # Try flat structure
        wavs = sorted(data_path.rglob("*.wav"))
        txts = sorted(data_path.rglob("*.txt"))
        pairs = list(zip(wavs, txts))
    else:
        wavs = sorted(audio_dir.glob("*.wav"))
        pairs = []
        for wav in wavs:
            txt = trans_dir / (wav.stem + ".txt")
            if txt.exists():
                pairs.append((wav, txt))

    print(f"  [{speaker_name}] Found {len(pairs)} audio/transcript pairs")

    for wav_path, txt_path in pairs:
        try:
            waveform, sr = torchaudio.load(str(wav_path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            duration = waveform.shape[1] / sr
            if duration < 0.5 or duration > 25.0:
                continue

            if sr != TARGET_SR:
                waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                continue

            out_path = processed_dir / f"{speaker_name}_{wav_path.stem}.wav"
            torchaudio.save(str(out_path), waveform, TARGET_SR)

            manifest_entries.append({
                "audio_path": str(out_path),
                "text": text,
                "duration_s": waveform.shape[1] / TARGET_SR,
                "speaker": speaker_name,
            })
        except Exception as e:
            continue

# Save manifest
manifest_path = Path("processed/manifest.jsonl")
with open(manifest_path, "w", encoding="utf-8") as f:
    for entry in manifest_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

total_hours = sum(e["duration_s"] for e in manifest_entries) / 3600
print(f"\n  Total utterances: {len(manifest_entries)}")
print(f"  Total audio: {total_hours:.1f} hours")
print(f"  Manifest saved: {manifest_path}")


# ============================================================================
# STEP 2: Load model and configure training
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Loading model and configuring LoRA")
print("=" * 60)

from models import Model, MISO_TTS_8B_CONFIG
from generator import load_llama3_tokenizer, _load_model
from moshi_compat import patch_bitsandbytes_import_for_unquantized_layers
from training.train import setup_model_for_training, LoRALinear, get_cosine_schedule_with_warmup, save_checkpoint
from training.dataset import TamilTTSDataset, TamilTTSCollator

patch_bitsandbytes_import_for_unquantized_layers()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

print("  Loading MisoTTS 8B base model...")
model = _load_model("MisoLabs/MisoTTS", MISO_TTS_8B_CONFIG, device="cpu", dtype=dtype)

config = {
    "strategy": "lora",
    "lora_rank": 32,
    "lora_alpha": 64.0,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_epochs": 10,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_steps": 100,
    "log_interval": 20,
    "gradient_checkpointing": True,
}

print("  Configuring LoRA training...")
model, trainable_params = setup_model_for_training(model, config)
model = model.to(device)


# ============================================================================
# STEP 3: Setup data pipeline
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: Setting up data pipeline")
print("=" * 60)

text_tokenizer = load_llama3_tokenizer()

from moshi.models import loaders
from huggingface_hub import hf_hub_download
mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device=device)
mimi.set_num_codebooks(MISO_TTS_8B_CONFIG.audio_num_codebooks)
mimi.eval()

dataset = TamilTTSDataset(
    manifest_path=str(manifest_path),
    text_tokenizer=text_tokenizer,
    audio_tokenizer=mimi,
    num_codebooks=MISO_TTS_8B_CONFIG.audio_num_codebooks,
)

collator = TamilTTSCollator(
    text_tokenizer=text_tokenizer,
    audio_tokenizer=mimi,
    num_codebooks=MISO_TTS_8B_CONFIG.audio_num_codebooks,
    device=device,
)

from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=2,
    collate_fn=collator,
    drop_last=True,
)

print(f"  Dataset: {len(dataset)} utterances")
print(f"  Batches per epoch: {len(dataloader)}")


# ============================================================================
# STEP 4: Training loop
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Training")
print("=" * 60)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=config["learning_rate"],
    weight_decay=config["weight_decay"],
    betas=(0.9, 0.95),
)

num_epochs = config["num_epochs"]
num_training_steps = len(dataloader) * num_epochs // config["gradient_accumulation_steps"]
scheduler = get_cosine_schedule_with_warmup(optimizer, config["warmup_steps"], num_training_steps)

output_dir = Path("outputs/tamil-kaggle-lora")
output_dir.mkdir(parents=True, exist_ok=True)

global_step = 0
best_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    epoch_start = time.time()

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

        loss_scaled = loss / config["gradient_accumulation_steps"]
        loss_scaled.backward()

        if (batch_idx + 1) % config["gradient_accumulation_steps"] == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                config["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        epoch_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % config["log_interval"] == 0:
            avg = epoch_loss / num_batches
            lr = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg:.4f} | LR: {lr:.2e}")

    avg_loss = epoch_loss / max(num_batches, 1)
    elapsed = time.time() - epoch_start
    print(f"\n  Epoch {epoch} done | Loss: {avg_loss:.4f} | Time: {elapsed:.0f}s")

    if avg_loss < best_loss:
        best_loss = avg_loss
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, config, output_dir)
        print(f"  Best model saved (loss={best_loss:.4f})")

print(f"\nTraining complete! Best loss: {best_loss:.4f}")
print(f"Model saved to: {output_dir}")


# ============================================================================
# STEP 5: Merge LoRA and test inference
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: Merging LoRA weights and testing")
print("=" * 60)

from training.train import merge_lora_weights
merge_lora_weights(model)

# Save merged model
try:
    from safetensors.torch import save_file
    save_file(model.state_dict(), str(output_dir / "model_merged.safetensors"))
except ImportError:
    torch.save(model.state_dict(), str(output_dir / "model_merged.pt"))

print("  Merged model saved")

# Quick inference test
print("\n  Testing inference...")
from generator import Generator
model.eval()
model.setup_caches(1)
gen = Generator(model)

test_text = "வணக்கம், இது தமிழ் பேச்சு தொகுப்பு."
print(f"  Generating: {test_text}")
try:
    audio = gen.generate(text=test_text, speaker=0, context=[], max_audio_length_ms=10000)
    torchaudio.save("tamil_test_output.wav", audio.unsqueeze(0).cpu(), gen.sample_rate)
    duration = audio.shape[0] / gen.sample_rate
    print(f"  Success! Generated {duration:.1f}s -> tamil_test_output.wav")
except Exception as e:
    print(f"  Inference error (expected on first run): {e}")

print("\n" + "=" * 60)
print("DONE! Download the model from outputs/tamil-kaggle-lora/")
print("=" * 60)
