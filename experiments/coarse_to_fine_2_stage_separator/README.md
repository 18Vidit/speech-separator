## Experiment : Coarse-to-Fine 2-Stage Separation

> **Status: Archived / Failed Experiment**  
> *This code is preserved for historical context and research documentation. It represents an earlier attempt at a 2-stage separation pipeline that was ultimately superseded by the Mamba-Dexformer architecture.*

## Concept Overview

*(See the included architecture diagram for a visual breakdown of this pipeline).*

This experiment tested a dual-stage extraction method based on 2022 speech separation literature (similar to the X-SepFormer concept). Instead of separating all speakers at once, it isolates them progressively:

1. **Stage 1 (Recursive OR-PIT):** Generates rough, imperfect speaker cues one by one.
2. **Stage 2 (Conditioned Extraction):** A cross-attention model that takes the original clean mixture and conditions it on the coarse cues from Stage 1 to extract the final, high-fidelity target speaker.

## Notable Engineering Features

While the model architecture didn't make the final cut, this script contains several advanced training mechanics:

* **Speaker Confusion Penalty:** Stage 2 utilizes a custom loss function that explicitly penalizes the model if its output sounds like a *different* real speaker in the mixture, rather than just optimizing for average reconstruction quality.
* **Curriculum Learning:** The model is trained progressively. It starts exclusively on 2-speaker mixtures (limiting the recursion unrolling) and slowly scales up to 5-speaker mixtures to stabilize learning.
* **Stateful Kaggle Resumability:** Custom-built to survive Kaggle's 9-hour session limits. It tracks epochs, optimizer states, and curriculum phase transitions in a local `run_state.json`. If a session dies, running the script again seamlessly resumes training from the exact drop-off point.

## Running the Archive Code

If you want to test or fork this specific pipeline:
1. Copy the contents of the python file into a single Kaggle Notebook cell.
2. Verify the `data_root` in the `CONFIG` dictionary matches your Kaggle dataset paths.
3. Run the cell. If the notebook hits a timeout, simply run the cell again to resume.