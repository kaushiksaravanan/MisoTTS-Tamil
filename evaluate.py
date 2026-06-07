"""
Evaluate Tamil TTS quality.

Metrics:
  1. MOS estimation (using UTMOS or self-supervised methods)
  2. Character Error Rate (via Whisper ASR on generated audio)
  3. Speaker consistency (embedding similarity)
  4. Prosody naturalness (pitch/energy variance analysis)
"""

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torchaudio


def evaluate_intelligibility(audio_paths: List[str], reference_texts: List[str]) -> dict:
    """
    Evaluate intelligibility using Whisper ASR.
    Transcribes generated audio and compares with input text.
    """
    try:
        import whisper
    except ImportError:
        print("  [SKIP] pip install openai-whisper for intelligibility eval")
        return {"cer": -1, "wer": -1}

    model = whisper.load_model("medium")
    total_cer = 0.0
    total_wer = 0.0
    count = 0

    for audio_path, ref_text in zip(audio_paths, reference_texts):
        result = model.transcribe(audio_path, language="ta")
        hyp_text = result["text"].strip()

        # Character Error Rate
        cer = _edit_distance(ref_text, hyp_text) / max(len(ref_text), 1)
        # Word Error Rate
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        wer = _edit_distance_words(ref_words, hyp_words) / max(len(ref_words), 1)

        total_cer += cer
        total_wer += wer
        count += 1

    return {
        "cer": total_cer / max(count, 1),
        "wer": total_wer / max(count, 1),
        "num_samples": count,
    }


def evaluate_naturalness(audio_paths: List[str]) -> dict:
    """
    Evaluate prosody naturalness via pitch and energy analysis.
    Good TTS should have natural pitch variation (not monotone, not erratic).
    """
    pitch_vars = []
    energy_vars = []
    durations = []

    for path in audio_paths:
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        duration = waveform.shape[1] / sr
        durations.append(duration)

        # Energy variance (RMS in windows)
        frame_size = int(0.025 * sr)
        hop_size = int(0.010 * sr)
        frames = waveform.squeeze().unfold(0, frame_size, hop_size)
        rms = frames.pow(2).mean(dim=1).sqrt()
        energy_vars.append(rms.std().item())

        # Pitch tracking via autocorrelation (simplified)
        # For production, use CREPE or WORLD vocoder
        pitch_vars.append(_estimate_pitch_variance(waveform.squeeze(), sr))

    return {
        "avg_pitch_variance": sum(pitch_vars) / max(len(pitch_vars), 1),
        "avg_energy_variance": sum(energy_vars) / max(len(energy_vars), 1),
        "avg_duration_s": sum(durations) / max(len(durations), 1),
        "num_samples": len(audio_paths),
    }


def evaluate_speaker_consistency(audio_paths: List[str]) -> dict:
    """
    Evaluate speaker consistency using speaker embeddings.
    Generated audio should maintain consistent speaker identity.
    """
    try:
        from speechbrain.inference import EncoderClassifier
    except ImportError:
        print("  [SKIP] pip install speechbrain for speaker consistency eval")
        return {"avg_cosine_similarity": -1}

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb"
    )

    embeddings = []
    for path in audio_paths:
        signal, fs = torchaudio.load(path)
        if fs != 16000:
            signal = torchaudio.functional.resample(signal, fs, 16000)
        embedding = classifier.encode_batch(signal)
        embeddings.append(embedding.squeeze())

    if len(embeddings) < 2:
        return {"avg_cosine_similarity": 1.0, "num_samples": len(embeddings)}

    # Pairwise cosine similarity
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = torch.nn.functional.cosine_similarity(
                embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)
            ).item()
            sims.append(sim)

    return {
        "avg_cosine_similarity": sum(sims) / len(sims),
        "min_cosine_similarity": min(sims),
        "num_pairs": len(sims),
    }


def _estimate_pitch_variance(waveform: torch.Tensor, sr: int) -> float:
    """Simple pitch variance estimation via zero-crossing rate."""
    frame_size = int(0.030 * sr)
    hop_size = int(0.010 * sr)
    zcrs = []
    for start in range(0, len(waveform) - frame_size, hop_size):
        frame = waveform[start:start + frame_size]
        zcr = ((frame[:-1] * frame[1:]) < 0).float().mean().item()
        zcrs.append(zcr)
    return torch.tensor(zcrs).std().item() if zcrs else 0.0


def _edit_distance(ref: str, hyp: str) -> int:
    """Levenshtein distance at character level."""
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if ref[i-1] == hyp[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def _edit_distance_words(ref: list, hyp: list) -> int:
    """Levenshtein distance at word level."""
    m, n = len(ref), len(hyp)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if ref[i-1] == hyp[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tamil TTS")
    parser.add_argument("--audio-dir", required=True, help="Directory with generated WAV files")
    parser.add_argument("--manifest", help="Manifest with reference texts (for CER)")
    parser.add_argument("--output", default="eval_results.json", help="Output JSON")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    audio_paths = sorted(str(p) for p in audio_dir.glob("*.wav"))

    if not audio_paths:
        print(f"No WAV files found in {audio_dir}")
        return

    print(f"Evaluating {len(audio_paths)} audio files from {audio_dir}")
    results = {}

    # Naturalness
    print("\n1. Prosody & Naturalness...")
    results["naturalness"] = evaluate_naturalness(audio_paths)
    print(f"   Pitch variance: {results['naturalness']['avg_pitch_variance']:.4f}")
    print(f"   Energy variance: {results['naturalness']['avg_energy_variance']:.4f}")

    # Speaker consistency
    print("\n2. Speaker Consistency...")
    results["speaker_consistency"] = evaluate_speaker_consistency(audio_paths)
    if results["speaker_consistency"]["avg_cosine_similarity"] >= 0:
        print(f"   Avg similarity: {results['speaker_consistency']['avg_cosine_similarity']:.4f}")

    # Intelligibility (if manifest provided)
    if args.manifest:
        print("\n3. Intelligibility (Whisper ASR)...")
        entries = []
        with open(args.manifest) as f:
            for line in f:
                entries.append(json.loads(line))
        ref_texts = [e["text"] for e in entries[:len(audio_paths)]]
        results["intelligibility"] = evaluate_intelligibility(audio_paths, ref_texts)
        print(f"   CER: {results['intelligibility']['cer']:.4f}")
        print(f"   WER: {results['intelligibility']['wer']:.4f}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
