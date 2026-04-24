# Dataset Recommendations for EEG Channel Reconstruction (N -> M)

## Goal Fit
Your roadmap targets channel reconstruction experiments such as 19 -> 5 -> 19, 19 -> 10 -> 19, and 19 -> 15 -> 19 with physics baselines and hybrid residual learning. Ideal datasets should have:
- Multi-channel EEG with known sensor labels and stable montages.
- Multiple subjects for LOSO validation.
- Enough recording duration to generate many fixed-length windows.
- Metadata for condition/task labels (optional but useful for downstream checks).

## Recommended Datasets (Priority Order)

## 1) TUH EEG Corpus (Temple University Hospital)
- Best for scale and robustness.
- Large number of subjects and sessions.
- Clinical recordings with variable conditions, useful for generalization tests.
- Good choice for proving model robustness and external validity.
- Notes:
  - Montage consistency varies across recordings.
  - You will need a channel harmonization step before reconstruction training.

## 2) CHB-MIT Scalp EEG Database
- Strong baseline for reproducible benchmarking.
- Multi-channel scalp EEG with many long recordings.
- Good for reconstruction + seizure-feature preservation experiments.
- Already partially present in this repo under MIT_DATA.
- Notes:
  - Pediatric epilepsy cohort, so conclusions should mention domain constraints.

## 3) PhysioNet EEG Motor Movement/Imagery Dataset (EEGMMI)
- Useful for task-based EEG and downstream validation.
- Multi-subject setup suitable for LOSO.
- Good for checking whether reconstructed channels preserve discriminative task features.
- Notes:
  - Ensure consistent preprocessing and channel mapping before comparisons.

## 4) BCI Competition IV 2a
- High-quality, commonly cited benchmark.
- 22 EEG channels and clean task protocol.
- Excellent for proving that reconstruction preserves motor imagery information.
- Notes:
  - Fewer channels than high-density datasets, but very paper-friendly and standardized.

## 5) OpenNeuro EEG studies (curated selection)
- Good source for external dataset testing and stress-testing generalization.
- Can provide diverse paradigms and recording conditions.
- Notes:
  - Pick studies with clear electrode metadata and enough subjects.

## Suggested Dataset Strategy for Your Paper

## Stage A (Fast prototyping)
- Start with CHB-MIT (already in repo context).
- Implement full end-to-end pipeline:
  - Subsampling.
  - Baselines (interpolation + physics).
  - ML and hybrid residual model.

## Stage B (Core validation)
- Add EEGMMI or BCI-IV-2a for standardized task-oriented validation.
- Run LOSO and report cross-subject stability.

## Stage C (Generalization claim)
- Add TUH subset as out-of-distribution validation.
- Emphasize performance drop patterns and robustness of hybrid model.

## Minimum Preprocessing Standard (for all datasets)
- Bandpass: 0.5 to 45 Hz.
- Notch: local powerline frequency (50 or 60 Hz).
- Re-reference: common average or linked mastoids (report choice explicitly).
- Channel harmonization: map all recordings to a fixed canonical channel set.
- Windowing: fixed windows (e.g., 2 to 5 s) with overlap for training samples.
- Split policy: subject-wise splits only (LOSO preferred).

## Why This Helps Your Roadmap
- Phase 1: Supports subject-wise robust preprocessing and LOSO.
- Phase 2: Enables fair baseline comparisons across methods.
- Phase 3: Provides enough samples to train residual models without overfitting to one subject.
- Phase 4+: Enables strong claims on time/frequency/spatial metrics and downstream usability.

## Practical First Choice
If you want the fastest credible start from your current repo, begin with CHB-MIT and then validate on EEGMMI.
