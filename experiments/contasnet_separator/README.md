## Experiment : Single-Stage Fixed-Output Conv-TasNet

> **Status: Archived / Failed Experiment**  
> *This code is preserved for historical context and research documentation. While Conv-TasNet is a highly effective architecture, this specific *single-stage* implementation designed to handle variable speaker counts (2-5) via a fixed 5-channel output proved significantly less stable and lower-performing than the Mamba-Dexformer approach.*

## Concept Overview

*(See the included architecture diagram for a detailed breakdown of this implementation).*

This experiment tested the viability of a classic, fully convolutional, single-stage pipeline (encoder-separator-decoder) working directly in the time domain.

The core design challenge addressed here was handling variable speaker counts (a mixture containing 3 speakers, followed by one containing 5) without changing the network structure.

### The Fixed-Output Mechanism

Unlike Experiment 1 (recursive) or Mamba (iterative), this model architecture was fixed at **5 output streams.** We managed variable counts using a "PIT-with-Silence" loss function:

1.  **Matched Active Speakers (PIT):** We use standard Permutation Invariant Training (Hungarian matching) to calculate the SI-SDR loss between the input's actual speakers and the model's *best matching* output streams.
2.  **Forced Inactive Streams:** Any output streams that were unassigned (e.g., outputs 4 and 5 in a 3-speaker mixture) are forced to absolute silence. They are penalized heavily via an L2 energy penalty if they contain any sound.

This mechanism was optimized specifically for a single GPU (lighter TCN blocks, allowing batch sizes of 8) and included full Kaggle stateful resumes (`run_state.json`).

## Running the Archive Code

If you want to test or fork this specific single-stage pipeline:

1.  Copy the contents of the python file into a single Kaggle Notebook cell.
2.  Verify the `data_root` in the `CONFIG` dictionary matches your Kaggle dataset paths.
3.  Run the cell. It includes the same 8-hour session management logic—run the cell again to resume training seamlessly.