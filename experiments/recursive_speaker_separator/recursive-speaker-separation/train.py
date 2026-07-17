"""
train.py  (v2 -- curriculum + weighted loss + larger batch)
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import MultiSpeakerMixtureDataset, variable_speaker_collate
from model import RecursiveSeparator
from losses import best_of_two_si_sdr_loss, stopping_bce_loss


def train_one_example(model, mixture, sources, max_iterations, device, stopping_loss_weight):
    current = mixture.unsqueeze(0).to(device)
    remaining = sources.unsqueeze(0).to(device)
    total_loss = 0.0
    n_steps = 0

    while remaining.shape[1] > 0 and n_steps < max_iterations:
        dominant, residual, stop_logit = model(current)
        step_loss, chosen_idx = best_of_two_si_sdr_loss(dominant, residual, remaining)
        total_loss = total_loss + step_loss

        k = chosen_idx.item()
        keep_mask = torch.ones(remaining.shape[1], dtype=torch.bool)
        keep_mask[k] = False
        remaining = remaining[:, keep_mask, :]

        has_speech_label = torch.tensor([1.0 if remaining.shape[1] > 0 else 0.0], device=device)
        total_loss = total_loss + stopping_loss_weight * stopping_bce_loss(stop_logit, has_speech_label)

        current = residual
        n_steps += 1

    return total_loss / max(n_steps, 1)


def build_stage_subset(dataset, max_speakers_this_stage):
    manifest_path = os.path.join(dataset.data_dir, "manifest.npy")
    keep_indices = []
    if os.path.exists(manifest_path):
        import numpy as np
        manifest = np.load(manifest_path, allow_pickle=True)
        manifest_by_idx = {m["idx"]: m["n_speakers"] for m in manifest}
        for i, idx in enumerate(dataset.indices):
            if manifest_by_idx.get(idx, 99) <= max_speakers_this_stage:
                keep_indices.append(i)
    else:
        for i in range(len(dataset)):
            if dataset[i]["n_speakers"] <= max_speakers_this_stage:
                keep_indices.append(i)
    return Subset(dataset, keep_indices)


def run_training_stage(model, optimizer, loader, epochs, max_iterations,
                        stopping_loss_weight, device, checkpoint_path,
                        start_epoch=0, stage_name=""):
    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"[{stage_name}] Epoch {epoch}")
        for batch in pbar:
            mixtures = batch["mixture"]
            sources_list = batch["sources_list"]
            optimizer.zero_grad()
            
            batch_loss_value = 0.0
            for b in range(mixtures.shape[0]):
                # 1. Calculate the loss for a single standalone example
                example_loss = train_one_example(
                    model, mixtures[b], sources_list[b], max_iterations, device, stopping_loss_weight,
                )
                
                # 2. Scale it down by the total batch size
                example_loss_scaled = example_loss / mixtures.shape[0]
                
                # 3. Backpropagate immediately! This drops the heavy graph cache from VRAM right away
                example_loss_scaled.backward()
                
                # 4. Log the tracking scalar safely
                batch_loss_value += example_loss.item()
                
            # 5. Clip and step gradients normally after all batch items have contributed
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += (batch_loss_value / mixtures.shape[0])
            n_batches += 1
            pbar.set_postfix({"loss": epoch_loss / n_batches})

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"[{stage_name}] Epoch {epoch}: avg loss = {avg_loss:.4f}")
        torch.save({
            "epoch": epoch, "stage": stage_name,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": avg_loss,
        }, checkpoint_path)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    full_dataset = MultiSpeakerMixtureDataset(args.data_dir)
    model = RecursiveSeparator(max_iterations=args.max_iterations, wavlm_model_name=args.wavlm_path).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "recursive_separator_v2.pt")
    start_epoch = 0
    start_stage_idx = 0
    stages = list(range(2, args.max_speaker_cap + 1))

    if os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        resumed_stage = ckpt.get("stage", f"stage_{stages[0]}")
        for i, s in enumerate(stages):
            if f"stage_{s}" == resumed_stage:
                start_stage_idx = i
                break
        if start_epoch >= args.epochs_per_stage:
            start_stage_idx += 1
            start_epoch = 0

    for stage_idx in range(start_stage_idx, len(stages)):
        max_speakers_this_stage = stages[stage_idx]
        stage_name = f"stage_{max_speakers_this_stage}"
        print(f"\n=== Curriculum stage: up to {max_speakers_this_stage} speakers ===")
        subset = build_stage_subset(full_dataset, max_speakers_this_stage)
        print(f"Stage dataset size: {len(subset)} examples")
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=variable_speaker_collate, num_workers=4)
        this_stage_start_epoch = start_epoch if stage_idx == start_stage_idx else 0
        run_training_stage(model, optimizer, loader, args.epochs_per_stage, args.max_iterations,
                            args.stopping_loss_weight, device, ckpt_path,
                            start_epoch=this_stage_start_epoch, stage_name=stage_name)

    print("\nCurriculum training complete.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--epochs_per_stage", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_iterations", type=int, default=6)
    ap.add_argument("--max_speaker_cap", type=int, default=5)
    ap.add_argument("--stopping_loss_weight", type=float, default=0.1)
    ap.add_argument("--wavlm_path", type=str, default="microsoft/wavlm-base-plus")
    args = ap.parse_args()
    train(args)