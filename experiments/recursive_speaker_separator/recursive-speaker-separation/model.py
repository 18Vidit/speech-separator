"""
model.py  (v2 -- fixed mask complementarity, WavLM feature-encoder frozen)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel


class WavLMFrontEnd(nn.Module):
    def __init__(self, model_name="microsoft/wavlm-base-plus", n_finetune_layers=2,
                 out_dim=256, local_files_only=None):
        super().__init__()
        if local_files_only is None:
            local_files_only = os.path.isdir(model_name)
        self.wavlm = WavLMModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.wavlm.freeze_feature_encoder()  # <-- fix: stop it trying to set requires_grad on non-leaf recursive inputs
        hidden_size = self.wavlm.config.hidden_size
        for p in self.wavlm.parameters():
            p.requires_grad = False
        total_layers = len(self.wavlm.encoder.layers)
        for layer in self.wavlm.encoder.layers[total_layers - n_finetune_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        self.proj = nn.Linear(hidden_size, out_dim)

    def forward(self, waveform):
        outputs = self.wavlm(waveform)
        hidden = outputs.last_hidden_state
        return self.proj(hidden)


class DualPathBlock(nn.Module):
    def __init__(self, dim, n_heads=4, ff_dim=512, dropout=0.1):
        super().__init__()
        self.intra = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.inter = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.norm_intra = nn.LayerNorm(dim)
        self.norm_inter = nn.LayerNorm(dim)

    def forward(self, x):
        B, n_chunks, chunk_size, dim = x.shape
        x_intra = x.reshape(B * n_chunks, chunk_size, dim)
        x_intra = self.intra(x_intra)
        x_intra = x_intra.reshape(B, n_chunks, chunk_size, dim)
        x = self.norm_intra(x + x_intra)
        x_inter = x.permute(0, 2, 1, 3).reshape(B * chunk_size, n_chunks, dim)
        x_inter = self.inter(x_inter)
        x_inter = x_inter.reshape(B, chunk_size, n_chunks, dim).permute(0, 2, 1, 3)
        x = self.norm_inter(x + x_inter)
        return x


class SepFormerSeparator(nn.Module):
    def __init__(self, encoder_dim=256, n_blocks=8, chunk_size=100, n_heads=4, ff_dim=512):
        super().__init__()
        self.chunk_size = chunk_size
        self.encoder_dim = encoder_dim
        self.encoder = nn.Conv1d(1, encoder_dim, kernel_size=32, stride=16, padding=8)
        self.blocks = nn.ModuleList([DualPathBlock(encoder_dim, n_heads=n_heads, ff_dim=ff_dim) for _ in range(n_blocks)])
        self.mask_head = nn.Linear(encoder_dim, 2 * encoder_dim)
        self.decoder = nn.ConvTranspose1d(encoder_dim, 1, kernel_size=32, stride=16, padding=8)

    def _chunk(self, x):
        B, T, dim = x.shape
        pad_len = (self.chunk_size - T % self.chunk_size) % self.chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        n_chunks = x.shape[1] // self.chunk_size
        return x.reshape(B, n_chunks, self.chunk_size, dim), pad_len

    def _unchunk(self, x, pad_len):
        B, n_chunks, chunk_size, dim = x.shape
        x = x.reshape(B, n_chunks * chunk_size, dim)
        if pad_len > 0:
            x = x[:, :-pad_len, :]
        return x

    def forward(self, waveform, film_params=None):
        B, T = waveform.shape
        x = waveform.unsqueeze(1)
        encoded = self.encoder(x)
        encoded = encoded.permute(0, 2, 1)

        if film_params is not None:
            gamma, beta = film_params.chunk(2, dim=-1)
            encoded = encoded * gamma.unsqueeze(1) + beta.unsqueeze(1)

        chunked, pad_len = self._chunk(encoded)
        for block in self.blocks:
            chunked = block(chunked)
        refined = self._unchunk(chunked, pad_len)

        raw_masks = self.mask_head(refined)
        B_, T_, _ = raw_masks.shape
        raw_masks = raw_masks.view(B_, T_, 2, self.encoder_dim)
        masks = torch.softmax(raw_masks, dim=2)
        mask_dom = masks[:, :, 0, :]
        mask_res = masks[:, :, 1, :]

        encoded_ct = encoded
        masked_dom = (encoded_ct * mask_dom).permute(0, 2, 1)
        masked_res = (encoded_ct * mask_res).permute(0, 2, 1)

        dominant = self.decoder(masked_dom).squeeze(1)
        residual = self.decoder(masked_res).squeeze(1)

        dominant = self._match_length(dominant, T)
        residual = self._match_length(residual, T)
        return dominant, residual

    @staticmethod
    def _match_length(x, target_len):
        if x.shape[-1] > target_len:
            return x[..., :target_len]
        elif x.shape[-1] < target_len:
            return F.pad(x, (0, target_len - x.shape[-1]))
        return x


class StoppingClassifier(nn.Module):
    def __init__(self, encoder_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, encoder_dim, kernel_size=32, stride=16, padding=8),
            nn.ReLU(),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(encoder_dim, 1)

    def forward(self, waveform):
        x = waveform.unsqueeze(1)
        feat = self.net(x).squeeze(-1)
        return self.classifier(feat).squeeze(-1)


class RecursiveSeparator(nn.Module):
    def __init__(self, wavlm_model_name="microsoft/wavlm-base-plus", encoder_dim=256, n_blocks=8, max_iterations=8, stopping_threshold=0.5):
        super().__init__()
        self.frontend = WavLMFrontEnd(wavlm_model_name, out_dim=encoder_dim)
        self.separator = SepFormerSeparator(encoder_dim=encoder_dim, n_blocks=n_blocks)
        self.stopper = StoppingClassifier()
        self.film = nn.Linear(encoder_dim, encoder_dim * 2)
        self.max_iterations = max_iterations
        self.stopping_threshold = stopping_threshold

    def _condition_and_separate(self, waveform):
        wavlm_feats = self.frontend(waveform)
        cond_vec = wavlm_feats.mean(dim=1)
        film_params = self.film(cond_vec)
        dominant, residual = self.separator(waveform, film_params=film_params)
        return dominant, residual

    def forward(self, waveform):
        dominant, residual = self._condition_and_separate(waveform)
        stop_logit = self.stopper(residual)
        return dominant, residual, stop_logit

    @torch.no_grad()
    def separate(self, waveform):
        self.eval()
        current = waveform.clone()
        separated = []
        for step in range(self.max_iterations):
            dominant, residual = self._condition_and_separate(current)
            separated.append(dominant.squeeze(0).cpu())
            stop_logit = self.stopper(residual)
            speech_prob = torch.sigmoid(stop_logit).item()
            if speech_prob < self.stopping_threshold:
                break
            current = residual
        return separated