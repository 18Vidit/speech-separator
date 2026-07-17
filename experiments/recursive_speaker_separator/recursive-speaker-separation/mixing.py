
import os
import argparse
import random
import glob
import numpy as np
import librosa
from scipy.signal import fftconvolve
from tqdm import tqdm

SR = 16000  # default; overridden by --sample_rate.
             # Use 8000 for the pretrained SepFormer (wsj02mix/wsj03mix) track --
             # those checkpoints were trained on 8kHz audio and expect 8kHz input.
             # Use 16000 for the WavLM/recursive track -- WavLM expects 16kHz.


def collect_speaker_files(speech_root):
    speaker_files = {}
    for root, _, files in os.walk(speech_root):
        for f in files:
            if f.endswith(".flac") or f.endswith(".wav"):
                speaker_id = root.split(os.sep)[-2]
                speaker_files.setdefault(speaker_id, []).append(os.path.join(root, f))
    return speaker_files


def load_clip(path, sr=None):
    audio, _ = librosa.load(path, sr=sr or SR)
    return audio


def normalize(audio, eps=1e-8):
    peak = np.max(np.abs(audio))
    if peak < eps:
        return audio
    return audio / peak


def fit_length(audio, length):
    if len(audio) < length:
        return np.pad(audio, (0, length - len(audio)))
    start = random.randint(0, len(audio) - length)
    return audio[start:start + length]


def apply_random_offset(audio, total_length):
    buf = np.zeros(total_length, dtype=np.float32)
    clip_len = min(len(audio), total_length)
    max_start = total_length - clip_len
    start = random.randint(0, max(max_start, 0))
    buf[start:start + clip_len] += audio[:clip_len]
    return buf


def convolve_rir(audio, rir_files):
    if not rir_files:
        return audio
    rir_path = random.choice(rir_files)
    rir, _ = librosa.load(rir_path, sr=SR)
    wet = fftconvolve(audio, rir)[:len(audio)]
    return normalize(wet) * np.max(np.abs(audio))


def add_background_noise(mixture, noise_files, snr_db_range=(5, 20)):
    if not noise_files:
        return mixture
    noise_path = random.choice(noise_files)
    noise, _ = librosa.load(noise_path, sr=SR)
    noise = fit_length(noise, len(mixture))

    sig_power = np.mean(mixture ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8
    snr_db = random.uniform(*snr_db_range)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)

    return mixture + noise


def make_mixture(speaker_ids, speaker_files, n_speakers, clip_samples,
                  rir_files=None, noise_files=None, use_random_offsets=True):
    chosen = random.sample(speaker_ids, n_speakers)
    clean_sources = []

    for spk in chosen:
        path = random.choice(speaker_files[spk])
        clip = normalize(load_clip(path))
        clip = fit_length(clip, clip_samples)

        if rir_files:
            clip = convolve_rir(clip, rir_files)

        gain = random.uniform(0.5, 1.0)
        clip = clip * gain

        if use_random_offsets:
            clip = apply_random_offset(clip, clip_samples)

        clean_sources.append(clip)

    mixture = np.sum(clean_sources, axis=0)

    if noise_files:
        mixture = add_background_noise(mixture, noise_files)

    peak = np.max(np.abs(mixture)) + 1e-8
    if peak > 1.0:
        scale = 1.0 / peak
        mixture = mixture * scale
        clean_sources = [s * scale for s in clean_sources]

    return mixture.astype(np.float32), np.stack(clean_sources).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=2000)
    ap.add_argument("--clip_seconds", type=float, default=4.0)
    ap.add_argument("--min_speakers", type=int, default=2)
    ap.add_argument("--max_speakers", type=int, default=5)
    ap.add_argument("--rir_root", default=None)
    ap.add_argument("--noise_root", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample_rate", type=int, default=16000,
                     help="8000 for the pretrained SepFormer track, 16000 for WavLM/recursive track.")
    ap.add_argument("--full_overlap", action="store_true",
                     help="Disable random start offsets -- matches original WSJ0-mix training distribution.")
    args = ap.parse_args()

    global SR
    SR = args.sample_rate

    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    clip_samples = int(args.clip_seconds * SR)

    print("Scanning speech files...")
    speaker_files = collect_speaker_files(args.speech_root)
    speaker_ids = [s for s, files in speaker_files.items() if len(files) > 0]
    print(f"Found {len(speaker_ids)} speakers")
    if len(speaker_ids) < args.max_speakers:
        raise ValueError("Not enough distinct speakers found for max_speakers setting. "
                          "Check --speech_root path.")

    rir_files = glob.glob(os.path.join(args.rir_root, "**", "*.wav"), recursive=True) if args.rir_root else None
    noise_files = glob.glob(os.path.join(args.noise_root, "**", "*.wav"), recursive=True) if args.noise_root else None
    if args.rir_root:
        print(f"Found {len(rir_files)} RIR files")
    if args.noise_root:
        print(f"Found {len(noise_files)} noise files")

    manifest = []
    for i in tqdm(range(args.n_samples), desc="Generating mixtures"):
        n_speakers = random.randint(args.min_speakers, args.max_speakers)
        mixture, sources = make_mixture(
            speaker_ids, speaker_files, n_speakers, clip_samples,
            rir_files=rir_files, noise_files=noise_files,
            use_random_offsets=not args.full_overlap,
        )
        mix_path = os.path.join(args.out_dir, f"mix_{i:06d}.npy")
        src_path = os.path.join(args.out_dir, f"sources_{i:06d}.npy")
        np.save(mix_path, mixture)
        np.save(src_path, sources)
        manifest.append({"idx": i, "n_speakers": n_speakers})

    manifest_path = os.path.join(args.out_dir, "manifest.npy")
    np.save(manifest_path, manifest)
    print(f"Done. Wrote {args.n_samples} mixtures to {args.out_dir}")


if __name__ == "__main__":
    main()