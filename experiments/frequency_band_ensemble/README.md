# Frequency-Band Specialist Ensemble (Blind Multi-Speaker Separation)

Blind separation pipeline for 2-5 concurrent speakers, no known speaker count at inference time.

**Architecture:** STFT -> split into 3 frequency bands -> per-band BiGRU specialist -> iterative
cross-band fusion (FiLM-conditioned) + residual refinement, repeated 3x -> per-band sigmoid mask
head -> mask applied to mixture spectrogram (phase reused) -> ISTFT.

**Speaker count**, which is unknown at inference, is predicted separately by a log-STFT + BiGRU +
attention-pooling classifier over {2,3,4,5}, which then routes the mixture to the matching
separation model (one independently-trained model per speaker count).

**Training:** PIT SI-SDR loss with a silence guard and a soft tanh score bound (avoids the
zero-gradient dead zone a hard clamp causes), AdamW with warmup + cosine decay, 40 epochs for
the separation models / 60 epochs for the count classifier.

**Evaluation** reports three numbers: Oracle SI-SDR (true count given, diagnostic only),
Conditional SI-SDR (only on correctly-counted examples), and Blind SI-SDR (every example,
misclassified count = 0 dB penalty) -- the last of these is the real reported score.

See `frequency_band_ensemble_notebook.ipynb` for the full implementation and
`architecture_diagram.jpg` for the block diagram.

## Results (test split)

**Oracle SI-SDR** — true speaker count given directly, separation quality only (not the real score):

| Speakers | Oracle SI-SDR |
|---|---|
| 2 | 12.26 dB |
| 3 | 11.25 dB |
| 4 | 4.59 dB |
| 5 | 6.58 dB |

**Speaker-count classifier accuracy** (test split, out of 200 examples per count):

| True count | Accuracy |
|---|---|
| 2 | 38.0% (76/200) |
| 3 | 36.0% (72/200) |
| 4 | 33.5% (67/200) |
| 5 | 36.5% (73/200) |

**Blind evaluation** — the real reported numbers, classifier + separation combined:

| Speakers | Conditional SI-SDR | Blind SI-SDR | Count accuracy |
|---|---|---|---|
| 2 | 13.73 dB | 5.36 dB | 39.0% |
| 3 | 8.66 dB | 2.94 dB | 34.0% |
| 4 | 6.82 dB | 2.22 dB | 32.5% |
| 5 | 1.14 dB | 0.42 dB | 37.0% |

**Takeaway:** separation quality (Oracle / Conditional) is reasonable for 2-3 speakers and drops
off for 4-5. The classifier is still only slightly above the 25% random-guess floor for 4 classes
even after the v4 changes (60 epochs, log-compressed input, attention pooling, hidden=192) — it
remains the main bottleneck holding down the Blind SI-SDR numbers, which is what a real deployment
would actually be scored on.
