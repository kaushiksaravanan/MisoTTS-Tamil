"""
Upload fine-tuned Tamil model to HuggingFace Hub.

Usage:
  python upload_to_hf.py --model-dir outputs/tamil-lora-v1 --repo-id kaushiksaravanan/MisoTTS-Tamil
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo


MODEL_CARD_TEMPLATE = """---
license: other
language:
  - ta
  - en
tags:
  - text-to-speech
  - tts
  - tamil
  - miso-tts
  - speech-synthesis
  - indic-languages
pipeline_tag: text-to-speech
datasets:
  - vickythefire2000/indic-tts-tamil-female
  - vickythefire2000/indic-tts-tamil-male
  - vickythefire2000/iisc-mile-tamil-asr-corpus
base_model: MisoLabs/MisoTTS
---

# MisoTTS Tamil 8B

State-of-the-art Tamil text-to-speech model, fine-tuned from [MisoTTS 8B](https://huggingface.co/MisoLabs/MisoTTS).

## Model Description

This is a LoRA fine-tuned version of MisoTTS 8B for Tamil (தமிழ்) speech synthesis.
It produces natural, expressive Tamil speech with proper prosody and pronunciation.

### Architecture
- **Backbone**: Llama 3.2 8B with LoRA (rank-32)
- **Decoder**: Llama 300M (full fine-tuned)
- **Audio Codec**: Mimi (32 codebooks)
- **Text Processing**: Tamil normalizer + optional ISO 15919 romanization

### Training Data
- Indic TTS Tamil Female (IIT Madras, studio quality)
- Indic TTS Tamil Male (IIT Madras, studio quality)
- IISc-MILE Tamil ASR Corpus (150+ hours, 531 speakers)

## Usage

```python
from infer_tamil import load_finetuned_model, generate_tamil_speech

generator = load_finetuned_model(model_dir="path/to/model", device="cuda")
audio = generate_tamil_speech(generator, text="வணக்கம், எப்படி இருக்கீங்க?", speaker=0)
```

## Limitations
- Trained primarily on read speech; conversational style may vary
- Works best with properly written Tamil text (not SMS abbreviations)
- Maximum output length ~60 seconds per generation

## Citation
If you use this model, please cite both MisoTTS and this fine-tune:
```
@misc{{misotts-tamil-2025,
  title={{MisoTTS Tamil: Fine-tuned 8B TTS for Tamil}},
  author={{Kaushik Saravanan}},
  year={{2025}},
  url={{https://github.com/kaushiksaravanan/MisoTTS-Tamil}}
}}
```
"""


def main():
    parser = argparse.ArgumentParser(description="Upload model to HuggingFace")
    parser.add_argument("--model-dir", required=True, help="Path to trained model directory")
    parser.add_argument("--repo-id", default="kaushiksaravanan/MisoTTS-Tamil", help="HF repo ID")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: {model_dir} does not exist")
        return

    api = HfApi()

    print(f"Creating/accessing repo: {args.repo_id}")
    create_repo(args.repo_id, exist_ok=True, private=args.private)

    # Write model card
    readme_path = model_dir / "README.md"
    readme_path.write_text(MODEL_CARD_TEMPLATE)

    # Upload files
    files_to_upload = []
    for pattern in ["*.safetensors", "*.pt", "config.json", "README.md"]:
        files_to_upload.extend(model_dir.glob(pattern))

    print(f"Uploading {len(files_to_upload)} files to {args.repo_id}...")
    for file_path in files_to_upload:
        print(f"  Uploading {file_path.name}...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=args.repo_id,
        )

    print(f"\nDone! Model available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
