
import os
import argparse
import numpy as np
import torch
from collections import defaultdict

from dataset import MultiSpeakerMixtureDataset
from model import RecursiveSeparator
from losses import si_sdr


def greedy_match_si_sdr(estimates, targets):
    targets = targets.clone()
    used = [False] * targets.shape[0]
    scores = []
    for est in estimates:
        best_score, best_idx = None, None
        for k in range(targets.shape[0]):
            if used[k]:
                continue
            score = si_sdr(est.unsqueeze(0), targets[k].unsqueeze(0)).item()
            if best_score is None or score > best_score:
                best_score, best_idx = score, k
        if best_idx is not None:
            used[best_idx] = True
            scores.append(best_score)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_iterations", type=int, default=8)
    ap.add_argument("--wavlm_path", type=str, default="microsoft/wavlm-base-plus")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RecursiveSeparator(max_iterations=args.max_iterations, wavlm_model_name=args.wavlm_path).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = MultiSpeakerMixtureDataset(args.data_dir)
    results_by_tier = defaultdict(list)
    speaker_count_errors = []

    for i in range(len(dataset)):
        sample = dataset[i]
        mixture = sample["mixture"].unsqueeze(0).to(device)
        sources = sample["sources"]
        n_true_speakers = sample["n_speakers"]

        separated = model.separate(mixture)
        n_estimated = len(separated)
        speaker_count_errors.append(abs(n_estimated - n_true_speakers))

        scores = greedy_match_si_sdr(separated, sources)
        if scores:
            results_by_tier[n_true_speakers].extend(scores)

        if i % 20 == 0:
            print(f"[{i}/{len(dataset)}] true={n_true_speakers} estimated={n_estimated} "
                  f"avg_si_sdr={np.mean(scores) if scores else float('nan'):.2f}")

    print("\n=== Results by speaker-count tier ===")
    for tier in sorted(results_by_tier.keys()):
        scores = results_by_tier[tier]
        print(f"{tier} speakers: mean SI-SDR = {np.mean(scores):.2f} dB (n={len(scores)} separated streams)")

    print(f"\nMean absolute speaker-count error: {np.mean(speaker_count_errors):.2f}")


if __name__ == "__main__":
    main()