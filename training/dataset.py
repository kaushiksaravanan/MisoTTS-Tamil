"""
PyTorch Dataset for MisoTTS Tamil fine-tuning.

Loads preprocessed manifest, tokenizes text with Llama 3.2 tokenizer,
encodes audio with Mimi codec, and produces training batches in the
format expected by Model.forward().
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from training.tamil_normalizer import normalize_tamil_text, is_tamil_text


class TamilTTSDataset(Dataset):
    """Dataset that lazily loads audio and produces tokenized frames."""

    def __init__(
        self,
        manifest_path: str,
        text_tokenizer,
        audio_tokenizer,
        num_codebooks: int = 32,
        max_audio_frames: int = 750,  # ~60s at 12.5Hz Mimi frame rate
        max_text_tokens: int = 256,
        speaker_map: Optional[dict] = None,
    ):
        self.entries = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                self.entries.append(entry)

        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.num_codebooks = num_codebooks
        self.max_audio_frames = max_audio_frames
        self.max_text_tokens = max_text_tokens

        if speaker_map is None:
            speakers = sorted(set(e["speaker"] for e in self.entries))
            self.speaker_map = {s: i for i, s in enumerate(speakers)}
        else:
            self.speaker_map = speaker_map

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]
        audio_path = entry["audio_path"]
        text = entry["text"]
        speaker = self.speaker_map.get(entry["speaker"], 0)

        # Normalize Tamil text
        if is_tamil_text(text):
            text = normalize_tamil_text(text)

        waveform, sr = torchaudio.load(audio_path)
        if sr != 24000:
            waveform = torchaudio.functional.resample(waveform, sr, 24000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return {
            "waveform": waveform,
            "text": text,
            "speaker": speaker,
        }


def collate_fn_with_tokenizers(batch, text_tokenizer, audio_tokenizer, num_codebooks=32, max_seq_len=2048, device="cpu"):
    """
    Collate a batch by encoding audio with Mimi and building the token/mask tensors.

    Returns dict with:
        tokens: (B, S, num_codebooks+1)
        tokens_mask: (B, S, num_codebooks+1)
        targets: (B, S, num_codebooks)
        targets_mask: (B, S, num_codebooks)
        decoder_idx: (B, S_amortized) - indices for amortized decoder
    """
    frame_size = num_codebooks + 1
    batch_tokens = []
    batch_masks = []

    for item in batch:
        waveform = item["waveform"].to(device)
        text = item["text"]
        speaker = item["speaker"]

        # Encode text
        text_with_speaker = f"[{speaker}] {text.lstrip()}"
        text_token_ids = text_tokenizer.encode(text_with_speaker)
        if len(text_token_ids) > 256:
            text_token_ids = text_token_ids[:256]

        text_frames = torch.zeros(len(text_token_ids), frame_size, dtype=torch.long)
        text_masks = torch.zeros(len(text_token_ids), frame_size, dtype=torch.bool)
        text_frames[:, -1] = torch.tensor(text_token_ids, dtype=torch.long)
        text_masks[:, -1] = True

        # Encode audio with Mimi
        with torch.no_grad():
            audio_codes = audio_tokenizer.encode(waveform.unsqueeze(0))[0]  # (K, T)

        # Add EOS frame
        eos = torch.zeros(audio_codes.shape[0], 1, device=audio_codes.device)
        audio_codes = torch.cat([audio_codes, eos], dim=1)
        num_audio_frames = audio_codes.shape[1]

        audio_frames = torch.zeros(num_audio_frames, frame_size, dtype=torch.long)
        audio_masks = torch.zeros(num_audio_frames, frame_size, dtype=torch.bool)
        audio_frames[:, :num_codebooks] = audio_codes.T.cpu().long()
        audio_masks[:, :num_codebooks] = True

        # Concatenate text + audio frames
        seq_tokens = torch.cat([text_frames, audio_frames], dim=0)
        seq_masks = torch.cat([text_masks, audio_masks], dim=0)

        # Truncate to max_seq_len
        if seq_tokens.shape[0] > max_seq_len:
            seq_tokens = seq_tokens[:max_seq_len]
            seq_masks = seq_masks[:max_seq_len]

        batch_tokens.append(seq_tokens)
        batch_masks.append(seq_masks)

    # Pad to same length
    max_len = max(t.shape[0] for t in batch_tokens)
    B = len(batch_tokens)

    tokens = torch.zeros(B, max_len, frame_size, dtype=torch.long)
    tokens_mask = torch.zeros(B, max_len, frame_size, dtype=torch.bool)

    for i, (t, m) in enumerate(zip(batch_tokens, batch_masks)):
        tokens[i, :t.shape[0]] = t
        tokens_mask[i, :m.shape[0]] = m

    # Build targets: shifted by 1 (next-frame prediction)
    targets = tokens[:, 1:, :num_codebooks].clone()
    targets_mask = tokens_mask[:, 1:, :num_codebooks].clone()

    # Input tokens (all but last frame)
    input_tokens = tokens[:, :-1]
    input_mask = tokens_mask[:, :-1]

    # Decoder index: all audio frame positions (for amortized training)
    # Use all positions where audio targets exist
    S = input_tokens.shape[1]
    audio_positions = targets_mask[:, :, 0]  # (B, S-1)
    # For simplicity in training, use all positions
    decoder_idx = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)

    return {
        "tokens": input_tokens,
        "tokens_mask": input_mask,
        "targets": targets,
        "targets_mask": targets_mask,
        "decoder_idx": decoder_idx,
    }


class TamilTTSCollator:
    """Stateful collator that holds references to tokenizers."""

    def __init__(self, text_tokenizer, audio_tokenizer, num_codebooks=32, max_seq_len=2048, device="cpu"):
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.num_codebooks = num_codebooks
        self.max_seq_len = max_seq_len
        self.device = device

    def __call__(self, batch):
        return collate_fn_with_tokenizers(
            batch, self.text_tokenizer, self.audio_tokenizer,
            self.num_codebooks, self.max_seq_len, self.device
        )
