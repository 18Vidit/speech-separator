# Mamba-Dexformer: Deflationary Extractor for Speech Separation

An efficient, lightweight speech separation engine utilizing Mamba State-Space Models (SSMs) to isolate variable numbers of overlapping speakers (up to 5) from a single audio mixture. 

Traditional speech separation models require memory-intensive dual-path architectures and fixed speaker outputs. **Mamba-Dexformer** solves this by combining the linear-time processing of Mamba with a **Deflationary Routing** strategy—extracting one dominant speaker at a time, calculating the OR-PIT loss, and routing the remaining audio into a residual track for iterative extraction.

## Performance Benchmarks

The model was evaluated using the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) metric. The lightweight configuration (4 Blocks, 256 Dimensions) achieved the following results across multi-speaker mixtures:

| Complexity | Average SI-SDR Score |
| :--- | :--- |
| **2 Speakers** | 3.06 dB |
| **3 Speakers** | 3.09 dB |
| **4 Speakers** | 3.47 dB |
| **5 Speakers** | 4.21 dB |

## Architecture Overview

The Mamba-Dexformer is trained from scratch and consists of three primary components:

1.  **1D Convolutional Encoder:** Downsamples the raw audio mixture waveform into a continuous 2D latent space representation (`d_model=256`). This compresses the temporal sequence, preventing memory bottlenecks.
2.  **Mamba Backbone (4 Blocks):** A stack of Unidirectional Selective State-Space Models. It processes the sequence with linear time complexity, tracking specific vocal pitches and acoustic features over long audio sequences without the quadratic memory constraints of standard Transformers.
3.  **Dual 1D ConvTranspose Decoders:** Upsamples the processed latent features back into two distinct physical audio waveforms:
    *   **Target Path:** The isolated dominant speaker.
    *   **Residual Path:** The remaining overlapping speakers and background noise.

## Repository Structure

```text
mamba-dexformer-speech-sep/
├── README.md                
├── requirements.txt         
├── app/                     
│   └── app.py               # Gradio frontend for the interactive web UI
├── src/                     
│   ├── model.py             # MambaDeflationaryExtractor architecture class
│   ├── evaluate.py          # SI-SDR metric and evaluation loop
│   └── generate_mix.py      # Dataset mixture generation logic
├── notebooks/               
│   ├── 01_data_prep.ipynb         # Raw data processing
│   ├── 02_training_pipeline.ipynb # Kaggle training pipeline & curriculum learning
│   └── 03_evaluation.ipynb        # Final benchmark scoring
├── weights/                 
│   └── mamba_dexformer_5spk_epoch_25.pth  # Final trained lightweight checkpoint
└── demo_audio/              
    ├── 5_speaker_mixture.wav
    └── separated_target.wav
```

## Getting Started

### 1. Installation
Clone the repository and install the strictly required dependencies (Note: Mamba requires specific CUDA bindings).
```bash
git clone https://github.com/18Vidit/speech-separator.git
cd speech-separator
pip install -r requirements.txt
```

### 2. Running the Interactive Demo
To launch the graphical web interface for uploading audio and testing the separation model:
```bash
python app/app.py
```
This will generate a local web address (e.g., `http://127.0.0.1:7860`). Open it in your browser to interact with the model.

### 3. Running CLI Evaluation
To evaluate the model against a test dataset via the command line:
```bash
python src/evaluate.py --data_dir path/to/dev_mix --checkpoint weights/mamba_dexformer_5spk_epoch_25.pth
```