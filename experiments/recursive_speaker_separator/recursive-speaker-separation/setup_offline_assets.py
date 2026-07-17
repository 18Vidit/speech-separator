"""
setup_offline_assets.py

One-time setup script that downloads everything the pipeline needs so it can
later run fully offline (e.g. on a Kaggle/Colab GPU session with internet
disabled, or on a machine without internet access on the training node).

This replaces the old "online-copy" notebook. It:
  1. Installs speechbrain + deps and caches their wheels locally so they can
     be reinstalled with `pip install --no-index` later.
  2. Downloads the two pretrained SepFormer checkpoints (2-speaker and
     3-speaker WSJ0-mix models) used as the finetuning starting point.
  3. Downloads the WavLM-base-plus checkpoint used as the frontend encoder
     for the recursive separator.

Run this once, with internet access, before running the offline
training/eval pipeline in run_pipeline.sh / the individual scripts.

Usage:
    python setup_offline_assets.py --out_dir ./offline_assets
"""

import argparse
import os
import shutil


def download_wheels(wheels_dir):
    os.makedirs(wheels_dir, exist_ok=True)
    packages = ["speechbrain", "hyperpyyaml", "sentencepiece", "ruamel.yaml<0.19.0"]
    for pkg in packages:
        cmd = f'pip download "{pkg}" -d "{wheels_dir}" --no-deps -q'
        print(f"[wheels] {cmd}")
        os.system(cmd)
    print(f"Wheels cached to {wheels_dir}")


def download_sepformer_checkpoints(out_dir):
    from speechbrain.inference.separation import SepformerSeparation

    two_spk_dir = os.path.join(out_dir, "sepformer_wsj02mix")
    three_spk_dir = os.path.join(out_dir, "sepformer_wsj03mix")

    SepformerSeparation.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir=two_spk_dir)
    SepformerSeparation.from_hparams(source="speechbrain/sepformer-wsj03mix", savedir=three_spk_dir)

    # Drop the speechbrain fetch cache -- it's not needed once files are local.
    shutil.rmtree(os.path.join(two_spk_dir, ".cache"), ignore_errors=True)
    shutil.rmtree(os.path.join(three_spk_dir, ".cache"), ignore_errors=True)
    print(f"SepFormer checkpoints saved to {two_spk_dir} and {three_spk_dir}")


def download_wavlm(out_dir):
    from transformers import WavLMModel

    wavlm_dir = os.path.join(out_dir, "wavlm-base-plus")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    model.save_pretrained(wavlm_dir)
    print(f"WavLM-base-plus saved to {wavlm_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./offline_assets",
                     help="Directory to store downloaded wheels/checkpoints.")
    ap.add_argument("--skip_wheels", action="store_true")
    ap.add_argument("--skip_sepformer", action="store_true")
    ap.add_argument("--skip_wavlm", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not args.skip_wheels:
        download_wheels(os.path.join(args.out_dir, "wheels"))
    if not args.skip_sepformer:
        download_sepformer_checkpoints(args.out_dir)
    if not args.skip_wavlm:
        download_wavlm(args.out_dir)

    print("\nDone. Point mixing/train/finetune scripts at the paths under "
          f"{args.out_dir} to run fully offline.")


if __name__ == "__main__":
    main()
