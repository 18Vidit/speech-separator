# Mamba-Dexformer: Deflationary Extractor for Speech Separation

An efficient, lightweight speech separation engine utilizing Mamba State-Space Models (SSMs) to isolate variable numbers of overlapping speakers (up to 5 concurrent voices) from a single audio mixture. 

Traditional speech separation models require memory-intensive dual-path architectures and rigid, fixed speaker outputs. **Mamba-Dexformer** solves this by combining the linear-time processing of Mamba with a **Deflationary Routing** strategy—extracting one dominant speaker at a time, calculating One-and-Rest Permutation Invariant Training (OR-PIT) loss, and routing the remaining audio into a residual track for iterative extraction.

---

## Evaluation Benchmarks (115 Epochs)

The generalized, fully trained model checkpoint (`mamba_dexformer_epoch_115.pth`) was systematically evaluated using the **Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)** metric across all multi-speaker validation configurations:

| Mixture Complexity | Average Evaluation Score (SI-SDR) |
| :--- | :--- |
| **2 Speakers** | 1.63 dB |
| **3 Speakers** | 3.38 dB |
| **4 Speakers** | 4.08 dB |
| **5 Speakers** | 4.21 dB |

### Important Finding: Catastrophic Forgetting & Shifting Residuals
During evaluation, a fascinating phenomenon was observed: the 2-speaker SI-SDR score degraded over the final training phases. Because this model uses a sequential curriculum learning pipeline (Phase 1: 2-spk -> Phase 2: 3-spk -> Phase 3: 4-spk -> Phase 4: 5-spk), the model experienced **Catastrophic Forgetting**.

Furthermore, the operational definition of the *Residual Decoder* shifts drastically across the curriculum:
* In the **2-speaker phase**, the residual decoder is trained to isolate **1 clean voice**.
* In the **5-speaker phase**, the residual decoder is optimized to output a dense, chaotic **4-speaker background pile**.

As a result, the final model over-processes simple 2-speaker mixtures, expecting a crowded background track, which highlights a core challenge in pure sequential training for deflationary speech architectures.

### Future Work to Fix This
To build a completely universal, scale-invariant separator, future iterations will implement **Joint Mixed-Batch Training** or **Dynamic Density Sampling**. Instead of progressing sequentially through speaker counts, the dataloader will randomly inject 2, 3, 4, and 5-speaker mixtures into *every single batch*. This will force the Mamba backbone and dual decoders to maintain generalized weights that dynamically adapt to the density of the input signal without abandoning past knowledge.

---

## Repository Structure

    mamba-dexformer-speech-sep/
    ├── README.md                # Project documentation and engineering analysis
    ├── requirements.txt         # Core dependencies (PyTorch, Mamba, Gradio)
    ├── app/                     
    │   └── app.py               # Gradio frontend web application 
    ├── src/                     
    │   ├── model.py             # MambaDeflationaryExtractor architecture definition
    │   ├── evaluate.py          # Validation benchmarking engine and SI-SDR loss
    │   └── inference.py         # Production deflationary loop for audio generation
    ├── notebooks/               
    │   ├── 01_data_prep.ipynb         # Dataset generation documentation (transparent LibriMix expansion)
    │   ├── 02_training_pipeline.ipynb # "From-scratch" clean curriculum training pipeline
    │   └── 03_evaluation.ipynb        # Benchmarking pipeline against multiple validation subsets
    ├── experiments/             
    │   └── [files]              # Archive of alternative model architectures and prototype scripts
    ├── weights/                 
    │   └── mamba_dexformer_epoch_115.pth  # Fully trained generalized 115-epoch model weights
    └── demo_audio/              
        ├── 2_speaker_mixture.wav
        ├── 3_speaker_mixture.wav
        ├── 4_speaker_mixture.wav
        └── 5_speaker_mixture.wav # Ready-to-use validation audio for the web demo

---

## Dataset Hosting

The custom evaluation dataset used for this project extends the standard LibriMix recipe to handle higher density scenarios (4 and 5 concurrent speakers), using `min mode` (trimming to the shortest utterance) and uniform loudness randomization. 

The complete generated dataset is hosted and available for download here:
**[Kaggle Dataset: speech-sep-mixtures](https://www.kaggle.com/datasets/viditarora18/speech-sep-mixtures)** 

*(Note: The repository notebooks are pre-configured to handle the nested directory structure of this specific Kaggle export out-of-the-box).*

---

## Technical Architecture Overview

1. **1D Convolutional Encoder:** Processes raw 16kHz audio waveforms into a continuous 2D latent space representation (d_model=256). This downsamples the raw signal, preventing memory bottlenecks.
2. **Mamba Backbone (4 Blocks):** A stack of Unidirectional Selective State-Space Models. It processes sequences with linear time complexity, tracking acoustic profiles and vocal pitches across long durations without the quadratic memory constraints of standard Transformers.
3. **Dual 1D ConvTranspose Decoders:** Upsamples latent features back into physical audio tracks:
    * **Target Path:** Isolates the dominant single speaker.
    * **Residual Path:** Aggregates remaining speakers for the next deflation pass.

---

## Deployment Guide

Because `mamba-ssm` relies heavily on native C++ compilation bindings and NVIDIA CUDA extensions, running it locally on Apple Silicon (Mac) or non-GPU environments will fail. It is highly recommended to run this repository inside an **NVIDIA GPU-accelerated container or Kaggle environment**.

### 1. Set Up a Virtual Environment (Recommended for GPU Machines)
```bash
# Clone the repository
git clone [https://github.com/18Vidit/speech-separator.git](https://github.com/18Vidit/speech-separator.git)
cd speech-separator

# Initialize and activate isolated environment
python3 -m venv venv
source venv/bin/activate

# Install torch ecosystem first to satisfy Mamba's build environment
pip install torch torchaudio numpy

# Install remaining dependencies without isolated build walls
pip install -r requirements.txt --no-build-isolation
```

### 2. Running on Kaggle (Easiest Cloud Setup)
To run the interactive web interface using Kaggle's free GPU accelerators, paste this block into a fresh Kaggle notebook cell (ensure **GPU T4 x2** or **P100** is activated in the settings panel):

```bash
!git clone https://github.com/18Vidit/speech-separator.git
%cd speech-separator
!pip install torch torchaudio numpy
!pip install -r requirements.txt --no-build-isolation
!python app/app.py
```

### 3. Launching the Interactive Web UI
Once the dependencies are installed, start the app:

```bash
python app/app.py
```

The application will launch and automatically output a public Gradio URL (e.g., https://xxxxxxxx.gradio.live). Open this URL in any browser to upload your audio mixtures and experience the live deflationary extraction!
