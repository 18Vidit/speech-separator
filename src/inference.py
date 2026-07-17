import os
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torchaudio
from src.model import MambaDeflationaryExtractor

def separate_audio_file(mixture_path, checkpoint_path, out_dir, max_speakers=5):
    """
    Extracts multiple speakers from a single audio file sequentially.
    This is the production inference engine.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}...")
    
    # 1. Initialize Model
    model = MambaDeflationaryExtractor(d_model=256, d_state=16, d_conv=4, expand=2, num_blocks=4)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Weights not found at {checkpoint_path}")
        
    # 2. Load Weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.to(device)
    model.eval()
    
    # 3. Process Audio
    mixture, sr = torchaudio.load(mixture_path)
    
    # Convert to mono if stereo
    if mixture.shape[0] > 1:
        mixture = mixture.mean(dim=0, keepdim=True)
        
    current_signal = mixture.unsqueeze(0).to(device) 
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Processing: {mixture_path}")
    
    extracted_files = []
    
    # 4. Deflationary Loop
    with torch.no_grad():
        for i in range(1, max_speakers + 1):
            est_speaker, est_residual = model(current_signal)
            
            speaker_audio = est_speaker.cpu() 
            out_path = os.path.join(out_dir, f"separated_speaker_{i}.wav")
            torchaudio.save(out_path, speaker_audio, sr)
            
            extracted_files.append(out_path)
            print(f"Extracted Speaker {i} -> {out_path}")
            
            # Feed the residual back into the model for the next speaker
            current_signal = est_residual.unsqueeze(1)

    print("Separation complete")
    return extracted_files # Returning the list of files makes the frontend UI easier to build