# EEG Feature Extraction Project Documentation

This repository contains scripts and modules designed to reconstruct source-level electroencephalography (EEG) data using varying methodologies—ranging from traditional biophysical forward-inverse solving over a head model (MNE), to hybrid Deep Learning ResNet models that learn to estimate data for dropped or missing channels.

## 📂 `scripts/data_creation/`

### `data_collection.py`
Connects to an active LSL (Lab Streaming Layer) stream to record live trial-based EEG data. By default, it acquires data from a 64-channel stream and records the samples to `.csv` files inside trial-specific subfolders.
- Wait for a key press (Spacebar), waits 10 seconds, collects 30-second blocks for `TRIAL_COUNT` trials.
- Ideal for extracting dataset blocks used for Baseline resting or stimulus-based measurements.

## 📂 `scripts/channel_analysis/`

### `channel_importance_analysis.py`
Evaluates and ranks EEG channels to determine which are strictly necessary to maintain a reliable source reconstruction mapping. To calculate "importance", it tests:
1. **Sensitvity:** Measuring the forward model spatial leadfield sensitivity.
2. **Variance:** Analyzing raw signal variance across recorded trials.
3. **Reconstruction Impact:** Progressively dropping variables and measuring Mean Squared Error impact on the spatial source mapping.

### `channel_reconstruction.py`
Compiles tests for channel space imputation. It performs side-by-side verification between:
1. **Biophysical Imputation:** Using MNE to source-localize the subset of available channels and then forward-projecting ("predicting") back into channel space to fill missing gaps.
2. **Fast ML Reconstruction:** Using a PyTorch `FastChannelReconstructor` Multi-Layer Perceptron (MLP). The network learns from normalized subset data and maps directly to full-channel predictions.

### `source_to_channel_model.py`
A generalized `SourceToChannelNet` ML architecture comparing prediction trajectories. It tests training an MLP directly on subsets of channels against training on actual Source Space network activations derived from `MNE dSPM` (dynamic statistical parametric mapping).

## 📂 `scripts/hybrid_ai_approach/`

### `augmented_approach.py`
The crowning piece showcasing an augmented physics-ML hybrid system.
- Converts the MNE biophysical channel projection base into a robust tensor representation. 
- Discovers the “residual differences” between the physics-based MNE guess for a dropped channel vs. the true signal.
- Feeds data into deep representation models (`EEGResidualNetSimple` [CNN] and `EEGResidualNet` [1D ResNet]).
- The model outputs an error correction value (the residual) which is then added to the MNE guess to synthesize highly accurate channel reconstructions.

### `residual_approach.py`
A lightweight precursor snippet highlighting the environment constants used to establish the physics model before routing the variables into optimizing loops.

## 📂 `scripts/biophysical_physics/`

### `reconstruct.py`
Processes batches of `.csv` recordings. Runs a strictly MNE-based pipeline leveraging a three-layer boundary element model (BEM: scalp, inner skull, outer skull). Generates high-fidelity visual plots capturing original vs. reconstructed accuracy metrics like Pearson's R and MSE. 

### `source_reconstruction.py`
Demonstration of obtaining brain source estimates (Source Space estimation) from single EEG files using typical parameters (Bandpass filtering `1-40 Hz`, Loose restrictions=0.2). Generates interactive 3D visualizations revealing internal brain activations.

## 📂 `scripts/tests_and_misc/`

### `sanity.py`
A unit-test style checker. Ensures the MNE forward/inverse architecture performs coherently on already provided known good channels (for example `Cz` correlation matches nearly 100% with reconstructions).

### `single_trial_test.py`
Ensures that MNE's `fsaverage` head configurations and BEM layouts are correctly loading. Generates a coregistered interactive 3D mesh highlighting electrodes on the skin/skull.

### `single_trial_animation.py`
Constructs real-time animated frames mapping magnitude Topomaps for the raw 19 channels as time progresses, compiling them into a GIF. Useful for data overview sanity checks.
