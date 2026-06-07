"""
Preprocess Tamil speech data for MisoTTS fine-tuning.

Steps:
  1. Discover audio/transcript pairs from downloaded datasets
  2. Resample audio to 24kHz (Mimi codec native rate)
  3. Encode audio with Mimi to get RVQ codes (32 codebooks)
  4. Tokenize Tamil text with Llama 3.2 tokenizer
  5. Write processed manifest (JSON lines) for training

Run: python prepare_data.py --data-dir data/ --output-dir processed/ [--num-workers 4]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import torch
import torchaudio


def find_pairs_indic_tts(data_dir: Path) -> list[dict]:
    """Find audio/transcript pairs in Indic TTS format."""
    pairs = []
    audio_dir = data_dir / "audio_files"
    trans_dir = data_dir / "trans_files"

    if not audio_dir.exists():
        audio_dir = data_dir / "wav"
        trans_dir = data_dir / "txt"

    if not audio_dir.exists():
        for wav in data_dir.rglob("*.wav"):
            txt = wav.with_suffix(".txt")
            if not txt.exists():
                parent = wav.parent.parent / "trans_files" / (wav.stem + ".txt")
                if parent.exists():
                    txt = parent
            if txt.exists():
                pairs.append({"audio": str(wav), "text": str(txt)})
        return pairs

    for wav in sorted(audio_dir.glob("*.wav")):
        txt = trans_dir / (wav.stem + ".txt")
        if txt.exists():
            pairs.append({"audio": str(wav), "text": str(txt)})
        else:
            stem_base = wav.stem.replace("train_tamilfem_", "").replace("train_tamilmale_", "")
            for candidate in trans_dir.glob(f"*{stem_base}*"):
                pairs.append({"audio": str(wav), "text": str(candidate)})
                break

    return pairs


def find_pairs_iisc_mile(data_dir: Path) -> list[dict]:
    """Find audio/transcript pairs in IISc-MILE format (Train/Test split with audio_files/ and trans_files/)."""
    pairs = []
    for split_dir in data_dir.iterdir():
        if not split_dir.is_dir():
            continue
        audio_dir = split_dir / "audio_files"
        trans_dir = split_dir / "trans_files"
        if audio_dir.exists() and trans_dir.exists():
            for wav in sorted(audio_dir.glob("*.wav")):
                txt = trans_dir / (wav.stem + ".txt")
                if txt.exists():
                    pairs.append({"audio": str(wav), "text": str(txt)})
    return pairs


def find_pairs_common_voice(data_dir: Path) -> list[dict]:
    """Find audio/transcript pairs in Common Voice TSV format."""
    pairs = []
    clips_dir = data_dir / "clips"
    if not clips_dir.exists():
        clips_dir = data_dir

    for tsv_name in ["validated.tsv", "train.tsv", "test.tsv", "dev.tsv"]:
        tsv_path = data_dir / tsv_name
        if not tsv_path.exists():
            continue
        with open(tsv_path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            path_idx = header.index("path") if "path" in header else 1
            sentence_idx = header.index("sentence") if "sentence" in header else 2
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > max(path_idx, sentence_idx):
                    audio_path = clips_dir / parts[path_idx]
                    if audio_path.exists():
                        pairs.append({
                            "audio": str(audio_path),
                            "text_content": parts[sentence_idx],
                        })
    return pairs


def process_single_utterance(
    item: dict,
    output_dir: Path,
    target_sr: int = 24000,
    max_duration_s: float = 30.0,
    min_duration_s: float = 0.5,
) -> Optional[dict]:
    """Process a single audio/text pair. Returns manifest entry or None if filtered."""
    audio_path = Path(item["audio"])

    try:
        waveform, sr = torchaudio.load(str(audio_path))
    except Exception:
        return None

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    duration = waveform.shape[1] / sr
    if duration > max_duration_s or duration < min_duration_s:
        return None

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    if "text_content" in item:
        text = item["text_content"]
    else:
        txt_path = Path(item["text"])
        try:
            text = txt_path.read_text(encoding="utf-8").strip()
        except Exception:
            return None

    if not text:
        return None

    rel_stem = audio_path.stem
    out_path = output_dir / f"{rel_stem}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), waveform, target_sr)

    return {
        "audio_path": str(out_path),
        "text": text,
        "duration_s": waveform.shape[1] / target_sr,
        "speaker": item.get("speaker", "unknown"),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess Tamil TTS data")
    parser.add_argument("--data-dir", default="data", help="Raw data directory")
    parser.add_argument("--output-dir", default="processed", help="Output directory for processed data")
    parser.add_argument("--target-sr", type=int, default=24000, help="Target sample rate (Mimi native)")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Max utterance duration (seconds)")
    parser.add_argument("--min-duration", type=float, default=0.5, help="Min utterance duration (seconds)")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    speaker_id = 0

    # Indic TTS Female
    indic_fem_dir = data_dir / "indic-tts-female"
    if indic_fem_dir.exists():
        pairs = find_pairs_indic_tts(indic_fem_dir)
        for p in pairs:
            p["speaker"] = "tamil_female_0"
        all_pairs.extend(pairs)
        print(f"[indic-female] Found {len(pairs)} pairs")

    # Indic TTS Male
    indic_male_dir = data_dir / "indic-tts-male"
    if indic_male_dir.exists():
        pairs = find_pairs_indic_tts(indic_male_dir)
        for p in pairs:
            p["speaker"] = "tamil_male_0"
        all_pairs.extend(pairs)
        print(f"[indic-male] Found {len(pairs)} pairs")

    # IISc-MILE
    mile_dir = data_dir / "iisc-mile"
    if mile_dir.exists():
        pairs = find_pairs_iisc_mile(mile_dir)
        for p in pairs:
            p["speaker"] = "mile_multi"
        all_pairs.extend(pairs)
        print(f"[iisc-mile] Found {len(pairs)} pairs")

    # Common Voice Tamil
    cv_dir = data_dir / "common-voice-tamil"
    if cv_dir.exists():
        pairs = find_pairs_common_voice(cv_dir)
        for p in pairs:
            p["speaker"] = "cv_multi"
        all_pairs.extend(pairs)
        print(f"[common-voice] Found {len(pairs)} pairs")

    if not all_pairs:
        print("ERROR: No audio/transcript pairs found. Run download_data.py first.")
        sys.exit(1)

    print(f"\nTotal pairs to process: {len(all_pairs)}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"Target sample rate: {args.target_sr} Hz")
    print("Processing...")

    audio_out = output_dir / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    manifest = []
    processed = 0
    skipped = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                process_single_utterance, item, audio_out,
                args.target_sr, args.max_duration, args.min_duration
            ): item
            for item in all_pairs
        }

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                manifest.append(result)
                processed += 1
            else:
                skipped += 1

            if (processed + skipped) % 500 == 0:
                print(f"  Progress: {processed + skipped}/{len(all_pairs)} "
                      f"(processed={processed}, skipped={skipped})")

    manifest.sort(key=lambda x: x["audio_path"])

    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nDone!")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Manifest: {manifest_path}")

    total_hours = sum(e["duration_s"] for e in manifest) / 3600
    print(f"  Total audio: {total_hours:.1f} hours")

    speakers = set(e["speaker"] for e in manifest)
    print(f"  Speakers: {len(speakers)}")


if __name__ == "__main__":
    main()
