# MisoTTS Tamil Fine-tuning - Kaggle GPU
# Datasets pre-mounted at /kaggle/input/ (zero download)
import os, sys, time, json
from pathlib import Path

os.system("pip install -q torch==2.4.0 torchaudio==2.4.0 torchtune==0.4.0 torchao==0.9.0 transformers==4.49.0 tokenizers==0.21.0 huggingface_hub==0.28.1 moshi==0.2.2 safetensors pyyaml")
os.system("pip install -q 'silentcipher @ git+https://github.com/SesameAILabs/silentcipher@d46d7d0893a583d8968ab3a6626e2289faec9152'")

if not Path("MisoTTS-Tamil").exists():
    os.system("git clone https://github.com/kaushiksaravanan/MisoTTS-Tamil.git")
os.chdir("MisoTTS-Tamil")
sys.path.insert(0, ".")

import torch
import torchaudio
print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# STEP 1: Prepare data
print("\nSTEP 1: Preparing data")
KAGGLE_DIRS = [
    ("/kaggle/input/indic-tts-tamil-female", "female_0"),
    ("/kaggle/input/indic-tts-tamil-male", "male_0"),
]
processed_dir = Path("processed/audio")
processed_dir.mkdir(parents=True, exist_ok=True)
manifest_entries = []
TARGET_SR = 24000

for data_dir, speaker_name in KAGGLE_DIRS:
    dp = Path(data_dir)
    if not dp.exists():
        print(f"  [SKIP] {data_dir}")
        continue
    audio_dir = dp / "audio_files"
    trans_dir = dp / "trans_files"
    if not audio_dir.exists():
        audio_dir = dp
        trans_dir = dp
    wavs = sorted(audio_dir.glob("*.wav"))
    count = 0
    for wav_path in wavs:
        txt_path = trans_dir / (wav_path.stem + ".txt")
        if not txt_path.exists():
            continue
        try:
            waveform, sr = torchaudio.load(str(wav_path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            dur = waveform.shape[1] / sr
            if dur < 0.5 or dur > 25.0:
                continue
            if sr != TARGET_SR:
                waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            out_path = processed_dir / f"{speaker_name}_{wav_path.stem}.wav"
            torchaudio.save(str(out_path), waveform, TARGET_SR)
            manifest_entries.append({"audio_path": str(out_path), "text": text, "duration_s": waveform.shape[1]/TARGET_SR, "speaker": speaker_name})
            count += 1
        except Exception:
            continue
    print(f"  [{speaker_name}] {count} utterances")

manifest_path = Path("processed/manifest.jsonl")
with open(manifest_path, "w", encoding="utf-8") as f:
    for e in manifest_entries:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
total_hours = sum(e["duration_s"] for e in manifest_entries) / 3600
print(f"  Total: {len(manifest_entries)} utterances, {total_hours:.1f} hours")

# STEP 2: Load model
print("\nSTEP 2: Loading MisoTTS 8B")
from models import Model, MISO_TTS_8B_CONFIG
from generator import load_llama3_tokenizer, _load_model
from moshi_compat import patch_bitsandbytes_import_for_unquantized_layers
from training.train import setup_model_for_training, LoRALinear, get_cosine_schedule_with_warmup, save_checkpoint

patch_bitsandbytes_import_for_unquantized_layers()
device = "cuda"
dtype = torch.bfloat16
model = _load_model("MisoLabs/MisoTTS", MISO_TTS_8B_CONFIG, device="cpu", dtype=dtype)
config = {"strategy": "lora", "lora_rank": 32, "lora_alpha": 64.0, "batch_size": 1, "gradient_accumulation_steps": 8, "num_epochs": 10, "learning_rate": 2e-4, "weight_decay": 0.01, "max_grad_norm": 1.0, "warmup_steps": 100, "log_interval": 20}
model, _ = setup_model_for_training(model, config)
model = model.to(device)

# STEP 3: Data pipeline
print("\nSTEP 3: Data pipeline")
text_tokenizer = load_llama3_tokenizer()
from moshi.models import loaders
from huggingface_hub import hf_hub_download
from training.dataset import TamilTTSDataset, TamilTTSCollator
mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
mimi = loaders.get_mimi(mimi_weight, device=device)
mimi.set_num_codebooks(MISO_TTS_8B_CONFIG.audio_num_codebooks)
mimi.eval()
dataset = TamilTTSDataset(manifest_path=str(manifest_path), text_tokenizer=text_tokenizer, audio_tokenizer=mimi, num_codebooks=MISO_TTS_8B_CONFIG.audio_num_codebooks)
collator = TamilTTSCollator(text_tokenizer=text_tokenizer, audio_tokenizer=mimi, num_codebooks=MISO_TTS_8B_CONFIG.audio_num_codebooks, device=device)
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=collator, drop_last=True)
print(f"  {len(dataset)} utterances, {len(dataloader)} batches/epoch")

# STEP 4: Train
print("\nSTEP 4: Training")
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01, betas=(0.9, 0.95))
num_steps = len(dataloader) * config["num_epochs"] // config["gradient_accumulation_steps"]
scheduler = get_cosine_schedule_with_warmup(optimizer, config["warmup_steps"], num_steps)
output_dir = Path("/kaggle/working/tamil-lora")
output_dir.mkdir(parents=True, exist_ok=True)
global_step, best_loss = 0, float("inf")

for epoch in range(config["num_epochs"]):
    model.train()
    epoch_loss, nb = 0.0, 0
    t0 = time.time()
    optimizer.zero_grad()
    for bi, batch in enumerate(dataloader):
        tokens = batch["tokens"].to(device)
        tokens_mask = batch["tokens_mask"].to(device)
        targets = batch["targets"].to(device)
        targets_mask = batch["targets_mask"].to(device)
        decoder_idx = batch["decoder_idx"].to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            _, _, c0_loss, c1_loss, loss = model(tokens=tokens, tokens_mask=tokens_mask, targets=targets, targets_mask=targets_mask, decoder_idx=decoder_idx)
        (loss / config["gradient_accumulation_steps"]).backward()
        if (bi+1) % config["gradient_accumulation_steps"] == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(); global_step += 1
        epoch_loss += loss.item(); nb += 1
        if (bi+1) % 20 == 0:
            print(f"  E{epoch} B{bi+1}/{len(dataloader)} loss={epoch_loss/nb:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
    avg = epoch_loss / max(nb, 1)
    print(f"  Epoch {epoch} | loss={avg:.4f} | {time.time()-t0:.0f}s")
    if avg < best_loss:
        best_loss = avg
        save_checkpoint(model, optimizer, scheduler, epoch, global_step, config, output_dir)

# STEP 5: Merge LoRA
print("\nSTEP 5: Merging LoRA weights")
from training.train import merge_lora_weights
merge_lora_weights(model)
try:
    from safetensors.torch import save_file
    save_file(model.state_dict(), str(output_dir / "model_merged.safetensors"))
except ImportError:
    torch.save(model.state_dict(), str(output_dir / "model_merged.pt"))

model.eval()
model.setup_caches(1)
from generator import Generator
gen = Generator(model)
print("  Testing inference...")
try:
    audio = gen.generate(text="vanakkam eppadi irukkinga", speaker=0, context=[], max_audio_length_ms=10000)
    torchaudio.save("/kaggle/working/tamil_test.wav", audio.unsqueeze(0).cpu(), gen.sample_rate)
    print(f"  Generated {audio.shape[0]/gen.sample_rate:.1f}s audio")
except Exception as e:
    print(f"  Inference error: {e}")

print("\nDONE! Model saved to /kaggle/working/tamil-lora/")
