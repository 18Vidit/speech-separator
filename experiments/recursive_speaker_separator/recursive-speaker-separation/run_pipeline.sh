#!/usr/bin/env bash
# End-to-end pipeline: mix data -> finetune pretrained SepFormer baselines ->
# train the recursive WavLM separator -> evaluate everything.
#
# Configure the paths below (or export these env vars before running) to
# point at your own LibriSpeech-style data and offline asset directories
# produced by setup_offline_assets.py.
set -euo pipefail

SPEECH_ROOT="${SPEECH_ROOT:-./data/librispeech-clean}"
ASSETS_DIR="${ASSETS_DIR:-./offline_assets}"
WORK_DIR="${WORK_DIR:-./working}"

mkdir -p "$WORK_DIR/checkpoints"

# ---------------------------------------------------------------------------
# 1. Build fixed 2-speaker / 3-speaker mixtures for finetuning the pretrained
#    SepFormer baselines (8kHz, matches the original WSJ0-mix checkpoints).
# ---------------------------------------------------------------------------
python mixing.py --speech_root "$SPEECH_ROOT" --out_dir "$WORK_DIR/train_2spk" --n_samples 8000 --min_speakers 2 --max_speakers 2 --sample_rate 8000 --full_overlap
python mixing.py --speech_root "$SPEECH_ROOT" --out_dir "$WORK_DIR/val_2spk"   --n_samples 300  --min_speakers 2 --max_speakers 2 --sample_rate 8000 --full_overlap --seed 123
python mixing.py --speech_root "$SPEECH_ROOT" --out_dir "$WORK_DIR/test_2spk"  --n_samples 300  --min_speakers 2 --max_speakers 2 --sample_rate 8000 --full_overlap --seed 999

python mixing.py --speech_root "$SPEECH_ROOT" --out_dir "$WORK_DIR/train_3spk" --n_samples 8000 --min_speakers 3 --max_speakers 3 --sample_rate 8000 --full_overlap
python mixing.py --speech_root "$SPEECH_ROOT" --out_dir "$WORK_DIR/val_3spk"   --n_samples 300  --min_speakers 3 --max_speakers 3 --sample_rate 8000 --full_overlap --seed 123
python mixing.py --speech_root "$SPEECH_ROOT" --out_dir "$WORK_DIR/test_3spk"  --n_samples 300  --min_speakers 3 --max_speakers 3 --sample_rate 8000 --full_overlap --seed 999

# ---------------------------------------------------------------------------
# 2. Finetune the pretrained SepFormer checkpoints on our mixtures (baseline).
# ---------------------------------------------------------------------------
python finetune_pretrained.py \
  --pretrained_dir "$ASSETS_DIR/sepformer_wsj02mix" \
  --data_dir "$WORK_DIR/train_2spk" --val_data_dir "$WORK_DIR/val_2spk" \
  --checkpoint_out "$WORK_DIR/checkpoints/sepformer_2spk.pt" \
  --num_spks 2 --epochs 30 --batch_size 4 --grad_accum_steps 8 --freeze_encoder_decoder

python finetune_pretrained.py \
  --pretrained_dir "$ASSETS_DIR/sepformer_wsj03mix" \
  --data_dir "$WORK_DIR/train_3spk" --val_data_dir "$WORK_DIR/val_3spk" \
  --checkpoint_out "$WORK_DIR/checkpoints/sepformer_3spk.pt" \
  --num_spks 3 --epochs 30 --batch_size 4 --grad_accum_steps 8 --freeze_encoder_decoder

# ---------------------------------------------------------------------------
# 3. Build variable-speaker (2-5) mixtures for the recursive separator, and
#    evaluate the finetuned SepFormer baselines on the fixed-count sets.
# ---------------------------------------------------------------------------
python mixing.py --speech_root "$SPEECH_ROOT" --out_dir "$WORK_DIR/train_dataset" --n_samples 5000 --min_speakers 2 --max_speakers 5 --sample_rate 16000
python mixing.py --speech_root "$SPEECH_ROOT" --out_dir "$WORK_DIR/test_dataset"  --n_samples 300  --min_speakers 2 --max_speakers 5 --sample_rate 16000 --seed 999

python evaluate_pretrained.py --pretrained_dir "$ASSETS_DIR/sepformer_wsj02mix" --checkpoint "$WORK_DIR/checkpoints/sepformer_2spk_best.pt" --data_dir "$WORK_DIR/test_2spk" --num_spks 2
python evaluate_pretrained.py --pretrained_dir "$ASSETS_DIR/sepformer_wsj03mix" --checkpoint "$WORK_DIR/checkpoints/sepformer_3spk_best.pt" --data_dir "$WORK_DIR/test_3spk" --num_spks 3

# ---------------------------------------------------------------------------
# 4. Train the recursive WavLM + dual-path separator with speaker-count
#    curriculum (2 -> 5 speakers), then evaluate it.
# ---------------------------------------------------------------------------
python train.py \
  --data_dir "$WORK_DIR/train_dataset" --checkpoint_dir "$WORK_DIR/checkpoints" \
  --epochs_per_stage 15 --batch_size 32 --lr 1e-4 --max_iterations 6 \
  --wavlm_path "$ASSETS_DIR/wavlm-base-plus"

python evaluate.py \
  --checkpoint "$WORK_DIR/checkpoints/recursive_separator_v2.pt" \
  --data_dir "$WORK_DIR/test_dataset" \
  --wavlm_path "$ASSETS_DIR/wavlm-base-plus"

echo "Pipeline complete. Checkpoints and results are under $WORK_DIR."
