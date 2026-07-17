
import torch
import torch.nn.functional as F


def si_sdr(estimate, target, eps=1e-8):
    target = target - target.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)

    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    proj = (dot / target_energy) * target

    noise = estimate - proj
    ratio = torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)


def si_sdr_loss(estimate, target):
    return -si_sdr(estimate, target).mean()


def best_of_two_si_sdr_loss(dominant_est, residual_est, remaining_targets):
    B, K, T = remaining_targets.shape
    best_idx = torch.zeros(B, dtype=torch.long, device=dominant_est.device)
    best_loss_per_example = torch.full((B,), float("inf"), device=dominant_est.device)

    for k in range(K):
        candidate_target = remaining_targets[:, k, :]
        candidate_residual_target = remaining_targets.sum(dim=1) - candidate_target

        loss_dom = -si_sdr(dominant_est, candidate_target)
        loss_res = -si_sdr(residual_est, candidate_residual_target)
        total = loss_dom + loss_res

        improved = total < best_loss_per_example
        best_loss_per_example = torch.where(improved, total, best_loss_per_example)
        best_idx = torch.where(improved, torch.full_like(best_idx, k), best_idx)

    return best_loss_per_example.mean(), best_idx


def stopping_bce_loss(stop_logit, has_speech_label):
    return F.binary_cross_entropy_with_logits(stop_logit, has_speech_label)