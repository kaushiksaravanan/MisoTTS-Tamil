"""
ONNX/OpenVINO export for maximum CPU inference speed.

Exports the Turbo model to ONNX and optionally optimizes with OpenVINO.
ONNX Runtime with CPU execution provider is the fastest portable option.

With ONNX Runtime optimizations:
  - Graph optimization (constant folding, redundant node removal)
  - Operator fusion (MatMul+Add, attention fusion)
  - INT8 dynamic quantization inside ORT
  - Parallel execution of independent ops

Expected performance:
  - Turbo 150M + ONNX ORT: ~3-8ms per inference (20 text tokens)
  - Turbo 150M + OpenVINO: ~2-5ms per inference (20 text tokens)
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.turbo_model import MisoTTSTurbo, TurboConfig


def export_to_onnx(
    model: MisoTTSTurbo,
    output_path: str = "misotts_turbo_tamil.onnx",
    max_text_len: int = 256,
    opset_version: int = 17,
):
    """Export Turbo model to ONNX format."""
    model.eval()

    # Dummy inputs
    text_tokens = torch.randint(0, 1000, (1, 20))
    text_mask = torch.ones(1, 20, dtype=torch.bool)
    speaker_ids = torch.tensor([0])

    # Export
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        model,
        (text_tokens, text_mask, speaker_ids),
        output_path,
        input_names=["text_tokens", "text_mask", "speaker_ids"],
        output_names=["logits", "durations"],
        dynamic_axes={
            "text_tokens": {0: "batch", 1: "text_len"},
            "text_mask": {0: "batch", 1: "text_len"},
            "speaker_ids": {0: "batch"},
            "logits": {0: "batch", 1: "audio_len"},
            "durations": {0: "batch", 1: "text_len"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"  Exported: {Path(output_path).stat().st_size / 1e6:.1f} MB")
    return output_path


def optimize_onnx(input_path: str, output_path: Optional[str] = None):
    """Optimize ONNX model with ORT transformers optimizer."""
    try:
        from onnxruntime.transformers import optimizer
        output_path = output_path or input_path.replace(".onnx", "_optimized.onnx")
        opt_model = optimizer.optimize_model(
            input_path,
            model_type="bert",  # Generic transformer
            num_heads=12,
            hidden_size=768,
        )
        opt_model.save_model_to_file(output_path)
        print(f"  Optimized: {Path(output_path).stat().st_size / 1e6:.1f} MB")
        return output_path
    except ImportError:
        print("  onnxruntime-transformers not available, skipping optimization")
        return input_path


def quantize_onnx_dynamic(input_path: str, output_path: Optional[str] = None):
    """Apply dynamic INT8 quantization in ONNX Runtime."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        output_path = output_path or input_path.replace(".onnx", "_int8.onnx")
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QInt8,
        )
        print(f"  INT8 quantized: {Path(output_path).stat().st_size / 1e6:.1f} MB")
        return output_path
    except ImportError:
        print("  onnxruntime quantization not available")
        return input_path


class OnnxTurboInference:
    """Run MisoTTS Turbo via ONNX Runtime for maximum CPU speed."""

    def __init__(self, onnx_path: str, num_threads: Optional[int] = None):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("pip install onnxruntime")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if num_threads:
            sess_options.intra_op_num_threads = num_threads
            sess_options.inter_op_num_threads = min(4, num_threads)
        else:
            sess_options.intra_op_num_threads = os.cpu_count() or 4

        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        print(f"  ONNX Runtime loaded: {onnx_path}")
        print(f"  Threads: {sess_options.intra_op_num_threads}")

    def generate(self, text_token_ids: list, speaker_id: int = 0) -> np.ndarray:
        """
        Generate audio codes from text tokens.
        Returns: (T_audio, 32) numpy array of audio codes.
        """
        text_tokens = np.array([text_token_ids], dtype=np.int64)
        text_mask = np.ones_like(text_tokens, dtype=np.bool_)
        speaker_ids = np.array([speaker_id], dtype=np.int64)

        outputs = self.session.run(
            None,
            {
                "text_tokens": text_tokens,
                "text_mask": text_mask,
                "speaker_ids": speaker_ids,
            },
        )

        logits = outputs[0]  # (1, T_audio, 32, vocab_size)
        audio_codes = logits.argmax(axis=-1).squeeze(0)  # (T_audio, 32)
        return audio_codes

    def benchmark(self, text_token_ids: list, num_runs: int = 50, warmup: int = 10):
        """Benchmark ONNX inference speed."""
        import time

        for _ in range(warmup):
            self.generate(text_token_ids)

        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            codes = self.generate(text_token_ids)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg = sum(times) / len(times)
        p50 = sorted(times)[len(times) // 2]
        p95 = sorted(times)[int(len(times) * 0.95)]
        audio_frames = codes.shape[0]
        audio_ms = audio_frames * 80

        print(f"\n  ONNX Benchmark ({num_runs} runs, {len(text_token_ids)} tokens):")
        print(f"    Avg: {avg:.2f} ms | P50: {p50:.2f} ms | P95: {p95:.2f} ms")
        print(f"    Output: {audio_frames} frames = {audio_ms:.0f} ms audio")
        print(f"    RTF: {avg / audio_ms:.4f}x")
        if avg < 10:
            print(f"    *** SUB-10ms ACHIEVED! ***")

        return {"avg_ms": avg, "p50_ms": p50, "p95_ms": p95, "rtf": avg / audio_ms}


def export_openvino(onnx_path: str, output_dir: str = "openvino_model"):
    """Convert ONNX to OpenVINO IR for Intel CPU optimization."""
    try:
        from openvino.tools import mo
        from openvino.runtime import Core

        print(f"  Converting to OpenVINO IR...")
        model = mo.convert_model(onnx_path)
        from openvino.runtime import serialize
        serialize(model, f"{output_dir}/model.xml")
        print(f"  OpenVINO model saved to {output_dir}/")
        return f"{output_dir}/model.xml"
    except ImportError:
        print("  OpenVINO not available (pip install openvino-dev)")
        return None


# ============================================================================
# Full export pipeline
# ============================================================================

def full_export_pipeline(model_or_checkpoint: str = None):
    """Run complete export: PyTorch → ONNX → INT8 → benchmark."""
    print("=" * 60)
    print("  MisoTTS Turbo Export Pipeline")
    print("=" * 60)

    # Load or create model
    config = TurboConfig()
    model = MisoTTSTurbo(config)

    if model_or_checkpoint and Path(model_or_checkpoint).exists():
        state = torch.load(model_or_checkpoint, map_location="cpu")
        model.load_state_dict(state)
        print(f"  Loaded checkpoint: {model_or_checkpoint}")

    # Export ONNX
    onnx_path = export_to_onnx(model, "misotts_turbo_tamil.onnx")

    # Quantize
    int8_path = quantize_onnx_dynamic(onnx_path)

    # Benchmark
    print("\n  Running benchmark...")
    try:
        engine = OnnxTurboInference(int8_path)
        # Simulate Tamil sentence (20 tokens)
        dummy_tokens = list(range(100, 120))
        engine.benchmark(dummy_tokens)
    except ImportError:
        print("  Install onnxruntime to benchmark: pip install onnxruntime")

    print("\n" + "=" * 60)
    print("  Export complete!")
    print(f"  Files: {onnx_path}, {int8_path}")
    print("=" * 60)


if __name__ == "__main__":
    full_export_pipeline()
