
import os
import argparse
import torch
from torch.utils.data import DataLoader

from speechbrain.inference.separation import SepformerSeparation
from dataset import MultiSpeakerMixtureDataset, variable_speaker_collate
from finetune_pretrained import forward_separate, permutation_invariant_si_sdr_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--num_spks", type=int, required=True, choices=[2, 3])
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    abs_pretrained_path = os.path.abspath(args.pretrained_dir)
    local_save_dir = os.path.join("/kaggle/working/speechbrain_cache", os.path.basename(abs_pretrained_path))
    os.makedirs(local_save_dir, exist_ok=True)

    model = SepformerSeparation.from_hparams(
        source=abs_pretrained_path,
        savedir=local_save_dir,
        run_opts={"device": device},
    )
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.mods.load_state_dict(ckpt["model_state"])
    model.mods.eval()

    dataset = MultiSpeakerMixtureDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=variable_speaker_collate)

    all_si_sdr = []
    with torch.no_grad():
        for batch in loader:
            mixture = batch["mixture"].to(device)
            sources_list = batch["sources_list"]
            targets = torch.stack(sources_list, dim=0).to(device)
            est_source = forward_separate(model, mixture, args.num_spks)
            loss = permutation_invariant_si_sdr_loss(est_source, targets)
            all_si_sdr.append(-loss.item())

    mean_si_sdr = sum(all_si_sdr) / len(all_si_sdr)
    print(f"\n=== {args.num_spks}-speaker fine-tuned SepFormer: mean SI-SDR = {mean_si_sdr:.2f} dB (n={len(all_si_sdr)} batches) ===")


if __name__ == "__main__":
    main()