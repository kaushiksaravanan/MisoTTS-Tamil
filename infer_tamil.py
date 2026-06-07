"""
Tamil TTS inference with fine-tuned MisoTTS model.

Generates speech from Tamil text using the fine-tuned model.
Supports both standalone generation and context-aware (multi-turn) synthesis.

Usage:
  python infer_tamil.py --text "வணக்கம், எப்படி இருக்கீங்க?"
  python infer_tamil.py --text "நான் நலமாக இருக்கிறேன்" --speaker 1
  python infer_tamil.py --interactive
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["NO_TORCH_COMPILE"] = "1"

import torch
import torchaudio

from generator import Generator, Segment, load_llama3_tokenizer, DEFAULT_MISO_TTS_REPO_ID
from models import Model, MISO_TTS_8B_CONFIG


def load_finetuned_model(
    base_model: str = DEFAULT_MISO_TTS_REPO_ID,
    finetuned_path: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Generator:
    """Load the fine-tuned Tamil model."""
    from generator import _load_model

    if finetuned_path and Path(finetuned_path).exists():
        finetuned = Path(finetuned_path)

        # Check if it's a merged model
        if (finetuned / "model_merged.safetensors").exists():
            model = _load_model(str(finetuned / "model_merged.safetensors"), MISO_TTS_8B_CONFIG, device, dtype)
        elif (finetuned / "model_merged.pt").exists():
            model = _load_model(str(finetuned / "model_merged.pt"), MISO_TTS_8B_CONFIG, device, dtype)
        elif finetuned.suffix in (".safetensors", ".pt"):
            model = _load_model(str(finetuned), MISO_TTS_8B_CONFIG, device, dtype)
        else:
            # Load base + apply LoRA checkpoint
            model = _load_model(base_model, MISO_TTS_8B_CONFIG, device="cpu", dtype=dtype)
            trainable_path = finetuned / "trainable_params.pt"
            if trainable_path.exists():
                trainable_state = torch.load(trainable_path, map_location="cpu")
                # Apply trainable parameters
                model_state = model.state_dict()
                for key, value in trainable_state.items():
                    if key in model_state:
                        model_state[key] = value
                model.load_state_dict(model_state, strict=False)
                print(f"Loaded {len(trainable_state)} fine-tuned parameters from {trainable_path}")
            model = model.to(device=device)
    else:
        print(f"No fine-tuned model found at {finetuned_path}, using base model")
        model = _load_model(base_model, MISO_TTS_8B_CONFIG, device, dtype)

    return Generator(model)


def generate_tamil_speech(
    generator: Generator,
    text: str,
    speaker: int = 0,
    context: Optional[List[Segment]] = None,
    max_audio_length_ms: float = 30_000,
    temperature: float = 0.85,
    topk: int = 50,
) -> torch.Tensor:
    """Generate Tamil speech audio."""
    if context is None:
        context = []

    audio = generator.generate(
        text=text,
        speaker=speaker,
        context=context,
        max_audio_length_ms=max_audio_length_ms,
        temperature=temperature,
        topk=topk,
    )
    return audio


def interactive_mode(generator: Generator):
    """Interactive Tamil TTS mode."""
    print("\n" + "=" * 60)
    print("  MisoTTS Tamil - Interactive Mode")
    print("  Type Tamil text to generate speech.")
    print("  Commands: /speaker <id>, /save <path>, /quit")
    print("=" * 60 + "\n")

    speaker = 0
    context = []
    output_idx = 0

    while True:
        try:
            text = input(f"[Speaker {speaker}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text:
            continue

        if text.startswith("/"):
            parts = text.split()
            cmd = parts[0].lower()
            if cmd == "/quit":
                break
            elif cmd == "/speaker" and len(parts) > 1:
                speaker = int(parts[1])
                print(f"  Speaker set to {speaker}")
            elif cmd == "/clear":
                context = []
                print("  Context cleared")
            elif cmd == "/save" and len(parts) > 1:
                # Save all generated audio
                if context:
                    all_audio = torch.cat([seg.audio for seg in context], dim=0)
                    path = parts[1]
                    torchaudio.save(path, all_audio.unsqueeze(0).cpu(), generator.sample_rate)
                    print(f"  Saved to {path}")
            continue

        print(f"  Generating...")
        audio = generate_tamil_speech(
            generator, text, speaker=speaker, context=context
        )

        segment = Segment(text=text, speaker=speaker, audio=audio)
        context.append(segment)

        output_path = f"tamil_output_{output_idx:04d}.wav"
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), generator.sample_rate)
        duration = audio.shape[0] / generator.sample_rate
        print(f"  Generated {duration:.1f}s -> {output_path}")
        output_idx += 1


def main():
    parser = argparse.ArgumentParser(description="Tamil TTS with MisoTTS")
    parser.add_argument("--text", type=str, help="Tamil text to synthesize")
    parser.add_argument("--speaker", type=int, default=0, help="Speaker ID (0=female, 1=male)")
    parser.add_argument("--output", type=str, default="tamil_output.wav", help="Output WAV path")
    parser.add_argument("--model-dir", type=str, default="outputs/tamil-lora-v1",
                        help="Fine-tuned model directory")
    parser.add_argument("--base-model", type=str, default=DEFAULT_MISO_TTS_REPO_ID,
                        help="Base model repo ID or path")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading Tamil TTS model...")
    generator = load_finetuned_model(
        base_model=args.base_model,
        finetuned_path=args.model_dir,
        device=device,
    )

    if args.interactive:
        interactive_mode(generator)
    elif args.text:
        print(f"Generating: {args.text}")
        audio = generate_tamil_speech(
            generator, args.text,
            speaker=args.speaker,
            temperature=args.temperature,
            topk=args.topk,
        )
        torchaudio.save(args.output, audio.unsqueeze(0).cpu(), generator.sample_rate)
        duration = audio.shape[0] / generator.sample_rate
        print(f"Generated {duration:.1f}s audio -> {args.output}")
    else:
        # Demo with sample Tamil sentences
        demo_sentences = [
            ("வணக்கம், எப்படி இருக்கீங்க?", 0),  # Hello, how are you?
            ("நான் நலமாக இருக்கிறேன், நன்றி.", 1),  # I'm fine, thank you.
            ("இன்று வானிலை மிகவும் அழகாக இருக்கிறது.", 0),  # The weather is beautiful today.
            ("தமிழ் ஒரு பழமையான மொழி.", 1),  # Tamil is an ancient language.
        ]

        print(f"\nGenerating {len(demo_sentences)} demo sentences...")
        segments = []
        for text, speaker in demo_sentences:
            print(f"  [{speaker}] {text}")
            audio = generate_tamil_speech(
                generator, text, speaker=speaker, context=segments,
                temperature=args.temperature, topk=args.topk,
            )
            segments.append(Segment(text=text, speaker=speaker, audio=audio))

        all_audio = torch.cat([seg.audio for seg in segments], dim=0)
        torchaudio.save(args.output, all_audio.unsqueeze(0).cpu(), generator.sample_rate)
        total_duration = all_audio.shape[0] / generator.sample_rate
        print(f"\nGenerated {total_duration:.1f}s conversation -> {args.output}")


if __name__ == "__main__":
    main()
