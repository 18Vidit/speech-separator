
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiSpeakerMixtureDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.indices = sorted(
            int(os.path.basename(p).split("_")[1].split(".")[0])
            for p in glob.glob(os.path.join(data_dir, "mix_*.npy"))
        )
        if len(self.indices) == 0:
            raise ValueError(f"No mix_*.npy files found in {data_dir}. "
                              f"Did you run mixing.py first?")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        mixture = np.load(os.path.join(self.data_dir, f"mix_{idx:06d}.npy"))
        sources = np.load(os.path.join(self.data_dir, f"sources_{idx:06d}.npy"))
        return {
            "mixture": torch.from_numpy(mixture).float(),
            "sources": torch.from_numpy(sources).float(),
            "n_speakers": sources.shape[0],
        }


def variable_speaker_collate(batch):
    mixtures = torch.stack([b["mixture"] for b in batch], dim=0)
    sources_list = [b["sources"] for b in batch]
    n_speakers = [b["n_speakers"] for b in batch]
    return {"mixture": mixtures, "sources_list": sources_list, "n_speakers": n_speakers}