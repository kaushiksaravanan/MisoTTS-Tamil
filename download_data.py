"""
Download Tamil TTS training data from Kaggle.

Datasets:
  1. Indic TTS Tamil Female (~2.5GB) - Studio-quality single speaker, IIT Madras
  2. Indic TTS Tamil Male (~2.8GB) - Studio-quality single speaker, IIT Madras
  3. IISc-MILE Tamil ASR Corpus (~13GB) - 150+ hours, 531 speakers, diverse

Run: python download_data.py [--datasets all|indic|mile] [--output-dir data/]
Requires: KAGGLE_USERNAME and KAGGLE_KEY env vars or ~/.kaggle/kaggle.json
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


DATASETS = {
    "indic-female": "vickythefire2000/indic-tts-tamil-female",
    "indic-male": "vickythefire2000/indic-tts-tamil-male",
    "iisc-mile": "vickythefire2000/iisc-mile-tamil-asr-corpus",
    "common-voice": "pylasandeep52/common-voice-corpus-21-09-tamil",
}


def download_dataset(slug: str, output_dir: Path):
    """Download and extract a Kaggle dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = slug.split("/")[1]
    zip_path = output_dir / f"{dataset_name}.zip"

    if any(output_dir.glob("**/*.wav")) or any(output_dir.glob("**/*.mp3")):
        print(f"  [SKIP] {slug} already extracted in {output_dir}")
        return

    print(f"  Downloading {slug} -> {output_dir}")
    cmd = [
        sys.executable, "-m", "kaggle", "datasets", "download",
        slug, "-p", str(output_dir)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] Download failed: {result.stderr}")
        return

    if zip_path.exists():
        print(f"  Extracting {zip_path.name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(output_dir)
            zip_path.unlink()
            print(f"  [OK] Extracted to {output_dir}")
        except zipfile.BadZipFile:
            print(f"  [ERROR] Corrupt zip file. Delete and retry: {zip_path}")
    else:
        print(f"  [OK] Downloaded (no zip found, files may be direct)")


def main():
    parser = argparse.ArgumentParser(description="Download Tamil TTS training data")
    parser.add_argument("--datasets", default="indic", choices=["all", "indic", "mile", "cv"],
                        help="Which datasets to download")
    parser.add_argument("--output-dir", default="data", help="Base output directory")
    args = parser.parse_args()

    base = Path(args.output_dir)

    targets = []
    if args.datasets in ("all", "indic"):
        targets.append(("indic-female", base / "indic-tts-female"))
        targets.append(("indic-male", base / "indic-tts-male"))
    if args.datasets in ("all", "mile"):
        targets.append(("iisc-mile", base / "iisc-mile"))
    if args.datasets in ("all", "cv"):
        targets.append(("common-voice", base / "common-voice-tamil"))

    print(f"Downloading {len(targets)} dataset(s) to {base.resolve()}")
    print("=" * 60)

    for name, output in targets:
        slug = DATASETS[name]
        print(f"\n[{name}] {slug}")
        download_dataset(slug, output)

    print("\n" + "=" * 60)
    print("Done. Run `python prepare_data.py` next to preprocess.")


if __name__ == "__main__":
    main()
