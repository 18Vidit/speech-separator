
import os
import argparse
import itertools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from speechbrain.inference.separation import SepformerSeparation
from dataset import MultiSpeakerMixtureDataset, variable_speaker_collate
from losses import si_sdr


def permutation_invariant_si_sdr_loss(est_source, targets):
    B, T, num_spks = est_source.shape
    est = est_source.permute(0, 2, 1)
    best_loss = torch.full((B,), float("inf"), device=est.device)
    for perm in itertools.permutations(range(num_spks)):
        permuted_targets = targets[:, list(perm), :]
        per_speaker_loss = -si_sdr(
            est.reshape(B * num_spks, T),
            permuted_targets.reshape(B * num_spks, T),
        ).reshape(B, num_spks).mean(dim=1)
        best_loss = torch.minimum(best_loss, per_speaker_loss)
    return best_loss.mean()


def forward_separate(model, mixture, num_spks):
    mix_w = model.mods.encoder(mixture)
    est_mask = model.mods.masknet(mix_w)
    mix_w_stack = torch.stack([mix_w] * num_spks)
    sep_h = mix_w_stack * est_mask
    est_source = torch.cat(
        [model.mods.decoder(sep_h[i]).unsqueeze(-1) for i in range(num_spks)], dim=-1,
    )
    T_origin = mixture.size(1)
    T_est = est_source.size(1)
    if T_origin > T_est:
        est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
    else:
        est_source = est_source[:, :T_origin, :]
    return est_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_dir", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--val_data_dir", default=None)
    ap.add_argument("--checkpoint_out", required=True)
    ap.add_argument("--num_spks", type=int, required=True, choices=[2, 3])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--grad_accum_steps", type=int, default=8,
                     help="Effective batch size = batch_size * grad_accum_steps.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--freeze_encoder_decoder", action="store_true")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading pretrained model from {args.pretrained_dir} (local, no internet needed)...")
    model = SepformerSeparation.from_hparams(
        source=args.pretrained_dir,
        savedir=args.pretrained_dir,
        run_opts={"device": device},
        freeze_params=False,
    )
    model.mods.train()

    if args.freeze_encoder_decoder:
        for p in model.mods.encoder.parameters():
            p.requires_grad = False
        for p in model.mods.decoder.parameters():
            p.requires_grad = False
        print("Encoder and decoder frozen -- only fine-tuning the masknet.")

    trainable_params = [p for p in model.mods.parameters() if p.requires_grad]
    print(f"Trainable parameter tensors: {len(trainable_params)}")

    dataset = MultiSpeakerMixtureDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         collate_fn=variable_speaker_collate, num_workers=4)

    val_loader = None
    if args.val_data_dir:
        val_dataset = MultiSpeakerMixtureDataset(args.val_data_dir)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=variable_speaker_collate, num_workers=2)

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler(enabled=use_amp)
    amp_device_type = "cuda" if use_amp else "cpu"

    os.makedirs(os.path.dirname(args.checkpoint_out), exist_ok=True)
    best_ckpt_path = args.checkpoint_out.replace(".pt", "_best.pt")
    start_epoch = 0
    best_val_loss = float("inf")

    if os.path.exists(args.checkpoint_out):
        print(f"Resuming from {args.checkpoint_out}")
        ckpt = torch.load(args.checkpoint_out, map_location=device)
        model.mods.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    for epoch in range(start_epoch, args.epochs):
        model.mods.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Fine-tune epoch {epoch}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            mixture = batch["mixture"].to(device)
            sources_list = batch["sources_list"]
            for s in sources_list:
                assert s.shape[0] == args.num_spks, (
                    f"Expected exactly {args.num_spks} speakers, got {s.shape[0]}."
                )
            targets = torch.stack(sources_list, dim=0).to(device)

            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                est_source = forward_separate(model, mixture, args.num_spks)
                loss = permutation_invariant_si_sdr_loss(est_source, targets)
                loss_scaled = loss / args.grad_accum_steps

            scaler.scale(loss_scaled).backward()

            is_accum_boundary = (step + 1) % args.grad_accum_steps == 0
            is_last_batch = (step + 1) == len(loader)
            if is_accum_boundary or is_last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": epoch_loss / n_batches})

        avg_train_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: avg PIT SI-SDR loss (train) = {avg_train_loss:.4f}")

        monitored_loss = avg_train_loss
        if val_loader is not None:
            model.mods.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    mixture = batch["mixture"].to(device)
                    targets = torch.stack(batch["sources_list"], dim=0).to(device)
                    est_source = forward_separate(model, mixture, args.num_spks)
                    val_loss = permutation_invariant_si_sdr_loss(est_source, targets)
                    val_loss_total += val_loss.item()
                    val_batches += 1
            monitored_loss = val_loss_total / max(val_batches, 1)
            print(f"Epoch {epoch}: avg PIT SI-SDR loss (val)   = {monitored_loss:.4f} (SI-SDR = {-monitored_loss:.2f} dB)")

        scheduler.step(monitored_loss)

        torch.save({
            "epoch": epoch, "model_state": model.mods.state_dict(),
            "optimizer_state": optimizer.state_dict(), "loss": avg_train_loss,
            "best_val_loss": best_val_loss, "num_spks": args.num_spks,
        }, args.checkpoint_out)

        if monitored_loss < best_val_loss:
            best_val_loss = monitored_loss
            torch.save({
                "epoch": epoch, "model_state": model.mods.state_dict(),
                "loss": monitored_loss, "num_spks": args.num_spks,
            }, best_ckpt_path)
            print(f"  -> new best checkpoint saved to {best_ckpt_path} (SI-SDR = {-monitored_loss:.2f} dB)")

    print("Fine-tuning complete.")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()