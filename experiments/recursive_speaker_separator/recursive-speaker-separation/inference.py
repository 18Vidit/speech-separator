
import os
import argparse
import torch
import librosa
import soundfile as sf

from model import RecursiveSeparator

SR = 16000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--input_wav", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_iterations", type=int, default=8)
    ap.add_argument("--stopping_threshold", type=float, default=0.5)
    ap.add_argument("--wavlm_path", type=str, default="microsoft/wavlm-base-plus")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    model = RecursiveSeparator(
        max_iterations=args.max_iterations,
        stopping_threshold=args.stopping_threshold,
        wavlm_model_name=args.wavlm_path,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    audio, _ = librosa.load(args.input_wav, sr=SR)
    waveform = torch.from_numpy(audio).float().unsqueeze(0).to(device)

    separated = model.separate(waveform)
    print(f"Found {len(separated)} speaker(s) in the mixture.")

    for i, speaker_audio in enumerate(separated):
        out_path = os.path.join(args.out_dir, f"speaker_{i + 1}.wav")
        sf.write(out_path, speaker_audio.numpy(), SR)
        print(f"  Saved {out_path}")


if __name__ == "__main__":
    main()