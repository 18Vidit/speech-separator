# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from mamba_ssm import Mamba

class MambaDeflationaryExtractor(nn.Module):
    def __init__(self, d_model=256, d_state=16, d_conv=4, expand=2, num_blocks=4):
        super().__init__()
        
        # Encoder: Audio waveform to latent features (1D Conv)
        self.encoder = nn.Conv1d(in_channels=1, out_channels=d_model, kernel_size=16, stride=8, padding=4)
        
        # Backbone: Stacked Mamba Blocks (Ultra-efficient sequential processing)
        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=d_model, # Model dimension
                d_state=d_state, # State dimension (controls memory compression)
                d_conv=d_conv,   # Local convolution width
                expand=expand    # Block expansion factor
            ) for _ in range(num_blocks)
        ])
        
        # Dual Decoders: 
        # Decoder A isolates the dominant speaker
        self.speaker_decoder = nn.ConvTranspose1d(in_channels=d_model, out_channels=1, kernel_size=16, stride=8, padding=4)
        # Decoder B outputs the rest of the mixture
        self.residual_decoder = nn.ConvTranspose1d(in_channels=d_model, out_channels=1, kernel_size=16, stride=8, padding=4)

    def forward(self, x):
        """
        x shape: (batch_size, 1, sequence_length)
        """
        # Encode
        encoded = self.encoder(x) # Shape: (batch, d_model, seq_len_latent)
        
        # Mamba expects shape (batch, seq_len, d_model), so we transpose
        latent = encoded.transpose(1, 2)
        
        # Pass through Mamba backbone
        for block in self.mamba_blocks:
            latent = block(latent)
            
        # Transpose back for decoding
        latent = latent.transpose(1, 2) # Shape: (batch, d_model, seq_len_latent)
        
        # Decode into two separate audio signals
        est_speaker = self.speaker_decoder(latent)
        est_residual = self.residual_decoder(latent)
        
        # Squeeze out the channel dimension for loss calculation
        return est_speaker.squeeze(1), est_residual.squeeze(1)