"""
MisoTTS Tamil Turbo: Distilled 150M model for real-time CPU inference.

The 8B model cannot achieve <10ms on CPU. Period. Physics wins.
But we can DISTILL its knowledge into a 150M model that CAN.

Architecture: Single-pass parallel decoder
  - No autoregressive backbone loop (biggest latency source)
  - Parallel codebook prediction (all 32 in one forward pass)
  - Duration predictor (avoids iterative length discovery)

Target: <10ms per frame on CPU = real-time streaming at 12.5Hz

Distillation strategy:
  1. Train 8B teacher on Tamil data (produces gold audio codes)
  2. Teacher generates (text → audio_codes) pairs for all training data
  3. Student (150M) trained on (text → audio_codes) with:
     - MSE loss on logits (knowledge distillation)
     - CE loss on hard targets (standard)
     - Duration prediction loss
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TurboConfig:
    """Configuration for the distilled Turbo model."""
    text_vocab_size: int = 128_256
    audio_vocab_size: int = 2051
    audio_num_codebooks: int = 32

    # Encoder (text → hidden)
    encoder_dim: int = 768
    encoder_layers: int = 6
    encoder_heads: int = 12
    encoder_ff_dim: int = 3072

    # Decoder (hidden → audio codes, NON-AUTOREGRESSIVE)
    decoder_dim: int = 512
    decoder_layers: int = 4
    decoder_heads: int = 8

    # Duration predictor
    duration_predictor_dim: int = 256
    duration_predictor_layers: int = 2

    max_text_len: int = 512
    max_audio_len: int = 1500  # ~2 min at 12.5Hz


class ConvPreNet(nn.Module):
    """Convolutional pre-net for text encoding."""

    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Conv1d(in_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D, T) -> conv -> (B, T, D)
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class DurationPredictor(nn.Module):
    """Predicts output frame count for each input text token."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Conv1d(in_d, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim) if i < num_layers - 1 else nn.Identity(),
            ])
        layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.ModuleList(layers)
        self.conv_layers = nn.Sequential(*layers[:-1])
        self.proj = layers[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = x.transpose(1, 2)  # (B, D, T)
        for i in range(0, len(self.layers) - 1, 3):
            h = self.layers[i](h)  # Conv
            h = self.layers[i + 1](h)  # ReLU
            # Skip LayerNorm on last
        h = h.transpose(1, 2)  # (B, T, D)
        durations = self.proj(h).squeeze(-1)  # (B, T)
        return F.softplus(durations)  # Positive durations


class LengthRegulator(nn.Module):
    """Expands text hidden states according to predicted durations."""

    def forward(self, h: torch.Tensor, durations: torch.Tensor, max_len: Optional[int] = None):
        """
        Args:
            h: (B, T_text, D) encoder hidden states
            durations: (B, T_text) integer durations per text token
        Returns:
            (B, T_audio, D) expanded hidden states
        """
        B, T, D = h.shape
        durations_int = durations.round().long().clamp(min=0)

        if max_len is None:
            max_len = durations_int.sum(dim=1).max().item()

        expanded = torch.zeros(B, max_len, D, device=h.device, dtype=h.dtype)

        for b in range(B):
            pos = 0
            for t in range(T):
                dur = durations_int[b, t].item()
                if pos + dur > max_len:
                    dur = max_len - pos
                if dur > 0:
                    expanded[b, pos:pos + dur] = h[b, t:t + 1].expand(dur, -1)
                pos += dur

        return expanded


class ParallelCodebookPredictor(nn.Module):
    """
    Predicts ALL 32 codebooks in PARALLEL (not autoregressively).
    This is the key innovation for <10ms inference.
    """

    def __init__(self, input_dim: int, num_codebooks: int, vocab_size: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.heads = nn.ModuleList([
            nn.Linear(input_dim, vocab_size) for _ in range(num_codebooks)
        ])

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, D) hidden states
        Returns:
            (B, T, num_codebooks, vocab_size) logits
        """
        logits = torch.stack([head(h) for head in self.heads], dim=2)
        return logits


class MisoTTSTurbo(nn.Module):
    """
    Distilled non-autoregressive TTS model.
    ~150M params, designed for <10ms/frame CPU inference.

    Architecture:
      Text → Encoder → Duration → Length Regulate → Parallel Decode → Audio Codes

    No autoregressive loop = no KV cache = no sequential bottleneck.
    """

    def __init__(self, config: TurboConfig):
        super().__init__()
        self.config = config

        # Text embedding
        self.text_embed = nn.Embedding(config.text_vocab_size, config.encoder_dim)
        self.pos_embed = nn.Embedding(config.max_text_len, config.encoder_dim)

        # Speaker embedding
        self.speaker_embed = nn.Embedding(16, config.encoder_dim)

        # Text encoder (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.encoder_dim,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_ff_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.encoder_layers)

        # Duration predictor
        self.duration_predictor = DurationPredictor(
            config.encoder_dim,
            config.duration_predictor_dim,
            config.duration_predictor_layers,
        )
        self.length_regulator = LengthRegulator()

        # Audio decoder (lightweight transformer)
        self.enc_to_dec = nn.Linear(config.encoder_dim, config.decoder_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.decoder_dim,
            nhead=config.decoder_heads,
            dim_feedforward=config.decoder_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.decoder_layers)

        # Parallel codebook prediction (all 32 at once!)
        self.codebook_predictor = ParallelCodebookPredictor(
            config.decoder_dim, config.audio_num_codebooks, config.audio_vocab_size
        )

        self._count_params()

    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"  MisoTTS Turbo: {total / 1e6:.1f}M parameters")

    def forward(
        self,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        speaker_ids: torch.Tensor,
        target_durations: Optional[torch.Tensor] = None,
        target_audio_codes: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            text_tokens: (B, T_text) token IDs
            text_mask: (B, T_text) attention mask
            speaker_ids: (B,) speaker IDs
            target_durations: (B, T_text) ground truth durations (training only)
            target_audio_codes: (B, T_audio, num_codebooks) GT codes (training only)
        """
        B, T = text_tokens.shape

        # Encode text
        positions = torch.arange(T, device=text_tokens.device).unsqueeze(0)
        h = self.text_embed(text_tokens) + self.pos_embed(positions)
        h = h + self.speaker_embed(speaker_ids).unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=~text_mask)

        # Predict durations
        pred_durations = self.duration_predictor(h)

        # Length regulation
        if target_durations is not None:
            # Training: use ground truth durations
            expanded = self.length_regulator(h, target_durations)
        else:
            # Inference: use predicted durations
            expanded = self.length_regulator(h, pred_durations)

        # Decode to audio codes (parallel!)
        dec_input = self.enc_to_dec(expanded)
        dec_output = self.decoder(dec_input)
        logits = self.codebook_predictor(dec_output)  # (B, T_audio, 32, 2051)

        result = {
            "logits": logits,
            "pred_durations": pred_durations,
        }

        # Compute losses if targets provided
        if target_audio_codes is not None:
            T_audio = min(logits.shape[1], target_audio_codes.shape[1])
            ce_loss = F.cross_entropy(
                logits[:, :T_audio].reshape(-1, self.config.audio_vocab_size),
                target_audio_codes[:, :T_audio].reshape(-1).long(),
                reduction="mean",
            )
            result["ce_loss"] = ce_loss

        if target_durations is not None:
            dur_loss = F.mse_loss(pred_durations, target_durations)
            result["dur_loss"] = dur_loss

        return result

    @torch.inference_mode()
    def generate(
        self,
        text_tokens: torch.Tensor,
        speaker_id: int = 0,
        duration_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Non-autoregressive generation. Single forward pass!
        Returns: (T_audio, num_codebooks) audio codes
        """
        B = 1
        text_mask = torch.ones(1, text_tokens.shape[1], dtype=torch.bool, device=text_tokens.device)
        speaker = torch.tensor([speaker_id], device=text_tokens.device)

        result = self.forward(text_tokens.unsqueeze(0), text_mask, speaker)
        logits = result["logits"]  # (1, T_audio, 32, 2051)

        # Greedy decode (or sample)
        audio_codes = logits.argmax(dim=-1).squeeze(0)  # (T_audio, 32)
        return audio_codes


# ============================================================================
# Distillation Training
# ============================================================================

class DistillationTrainer:
    """Train Turbo student from 8B teacher."""

    def __init__(
        self,
        teacher_generator,
        student: MisoTTSTurbo,
        text_tokenizer,
        audio_tokenizer,
        device: str = "cpu",
        temperature: float = 4.0,
        alpha_ce: float = 0.5,
        alpha_kd: float = 0.5,
    ):
        self.teacher = teacher_generator
        self.student = student
        self.text_tokenizer = text_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.device = device
        self.temperature = temperature
        self.alpha_ce = alpha_ce
        self.alpha_kd = alpha_kd

    def generate_teacher_targets(self, text: str, speaker: int) -> dict:
        """Generate gold audio codes from teacher model."""
        audio = self.teacher.generate(
            text=text, speaker=speaker, context=[],
            max_audio_length_ms=30_000, temperature=0.7, topk=50,
        )
        # Encode back to codes for distillation target
        with torch.no_grad():
            codes = self.audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        return {
            "audio_codes": codes.T,  # (T, K)
            "audio": audio,
        }

    def compute_loss(self, batch: dict) -> dict:
        """Compute distillation loss."""
        result = self.student(
            text_tokens=batch["text_tokens"],
            text_mask=batch["text_mask"],
            speaker_ids=batch["speaker_ids"],
            target_durations=batch.get("durations"),
            target_audio_codes=batch.get("audio_codes"),
        )

        losses = {}
        if "ce_loss" in result:
            losses["ce_loss"] = result["ce_loss"] * self.alpha_ce
        if "dur_loss" in result:
            losses["dur_loss"] = result["dur_loss"]

        losses["total"] = sum(losses.values())
        return losses


# ============================================================================
# Quick performance estimate
# ============================================================================

def estimate_turbo_latency():
    """Estimate MisoTTS Turbo latency on CPU."""
    config = TurboConfig()
    model = MisoTTSTurbo(config)
    model.eval()

    # Simulate 20-token input (typical short Tamil sentence)
    text = torch.randint(0, 1000, (1, 20))
    mask = torch.ones(1, 20, dtype=torch.bool)
    speaker = torch.tensor([0])

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(text, mask, speaker)

    # Benchmark
    import time
    times = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            result = model(text, mask, speaker)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)

    avg = sum(times) / len(times)
    print(f"\n  MisoTTS Turbo Latency (20 tokens):")
    print(f"    Forward pass: {avg:.1f} ms")
    print(f"    Output frames: ~{result['logits'].shape[1]}")
    print(f"    Audio duration: ~{result['logits'].shape[1] * 80:.0f} ms")
    print(f"    Real-time factor: {avg / (result['logits'].shape[1] * 80):.3f}x")

    return avg


if __name__ == "__main__":
    estimate_turbo_latency()
