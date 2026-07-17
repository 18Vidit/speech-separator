import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torchaudio
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader
from src.model import MambaDeflationaryExtractor

# 1. metrics and loss
def calc_si_sdr(estimate, target, epsilon=1e-8):
    target = target - torch.mean(target, dim=-1, keepdim=True)
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + epsilon
    dot_product = torch.sum(target * estimate, dim=-1, keepdim=True)
    alpha = dot_product / target_energy
    target_scaled = alpha * target
    noise = estimate - target_scaled
    signal_energy = torch.sum(target_scaled ** 2, dim=-1) + epsilon
    noise_energy = torch.sum(noise ** 2, dim=-1) + epsilon
    ratio = torch.clamp(signal_energy / noise_energy, min=epsilon)
    si_sdr = 10 * torch.log10(ratio)
    return -si_sdr

def or_pit_loss(est_speaker, est_residual, targets):
    batch_size = targets.shape[0]
    num_speakers = targets.shape[1] 
    
    target_speakers = torch.chunk(targets, num_speakers, dim=1)
    
    best_losses = []
    best_indices = []
    
    for b in range(batch_size):
        b_est_spk = est_speaker[b]
        b_est_res = est_residual[b]
        b_targets = [t[b].squeeze(0) for t in target_speakers]
        
        best_loss_tensor = None
        best_idx = 0
        
        for i in range(num_speakers):
            loss_spk = calc_si_sdr(b_est_spk.unsqueeze(0), b_targets[i].unsqueeze(0))
            res_target = sum([b_targets[j] for j in range(num_speakers) if j != i])
            loss_res = calc_si_sdr(b_est_res.unsqueeze(0), res_target.unsqueeze(0))
            
            total_loss = loss_spk + loss_res
            
            if best_loss_tensor is None or total_loss < best_loss_tensor:
                best_loss_tensor = total_loss
                best_idx = i
                
        best_losses.append(best_loss_tensor)
        best_indices.append(best_idx)
        
    return torch.stack(best_losses).mean(), best_indices

# 2. dataset loader
class CustomSpeechMixtureDataset(Dataset):
    def __init__(self, base_dir, num_speakers, fixed_length=80000): 
        self.base_dir = base_dir
        self.num_speakers = num_speakers
        self.fixed_length = fixed_length
        
        self.mix_folders = sorted([
            f for f in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('mix_')
        ])

    def __len__(self):
        return len(self.mix_folders)

    def __getitem__(self, idx):
        folder_name = self.mix_folders[idx]
        folder_path = os.path.join(self.base_dir, folder_name)
        
        mix_path = os.path.join(folder_path, "mix.wav")
        mixture, _ = torchaudio.load(mix_path)
        mixture = self._pad_or_truncate(mixture)
        
        sources = []
        for i in range(1, self.num_speakers + 1):
            source_path = os.path.join(folder_path, f"s{i}.wav")
            source_audio, _ = torchaudio.load(source_path)
            source_audio = self._pad_or_truncate(source_audio)
            sources.append(source_audio)
            
        target_sources = torch.cat(sources, dim=0)
        return mixture, target_sources

    def _pad_or_truncate(self, audio):
        length = audio.shape[-1]
        if length > self.fixed_length:
            return audio[:, :self.fixed_length]  
        else:
            padding = self.fixed_length - length
            return torch.nn.functional.pad(audio, (0, padding))

# 3. evaluation engine
def evaluate_validation_set(model, val_loader, device):
    """Evaluates the model against a validation dataset returning average SI-SDR."""
    model.eval()
    total_si_sdr = 0.0
    valid_batches = 0 
    
    with torch.no_grad(): 
        for mixture, targets in val_loader:
            mixture = mixture.to(device)
            targets = targets.to(device)
            
            est_speaker, est_residual = model(mixture)
            loss, _ = or_pit_loss(est_speaker, est_residual, targets)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            si_sdr_db = -1 * loss.item() 
            total_si_sdr += si_sdr_db
            valid_batches += 1
            
    if valid_batches == 0:
        return 0.0
        
    return total_si_sdr / valid_batches

