# EEG Channel Reconstruction Research Roadmap

## Phase 0: Problem Definition
- **Define input and output spaces**
  - **Input:** $X_K$ (subset of EEG channels, e.g., 5, 10, or 15)
  - **Output:** $\hat{X}_N$ (full EEG channels, e.g., 19 or 32)
- **Specify electrode layout**
  - Standard 10–20 or 10–10 system
- **Define reconstruction scenarios**
  - $19 \rightarrow 5 \rightarrow 19$
  - $19 \rightarrow 10 \rightarrow 19$
  - $19 \rightarrow 15 \rightarrow 19$
- **Define evaluation goals**
  - Signal fidelity
  - Frequency preservation
- **Hypotheses (Suggested Additions)**
  - *H1: Deep learning significantly outperforms Spline Interpolation for <10 channels.*
  - *H2: The Physics-AI Hybrid model will resolve the "spatial blurring" problem of pure MNE models.*

---

## 📊 Phase 1: Dataset & Preprocessing
### 1. Dataset Selection
- Use the base dataset provided in the directory (expand sequentially by adding future samples).

### 2. Preprocessing Pipeline
- Bandpass filter (0.5–45 Hz)
- Artifact handling (remove or standardize occurrences)
- Re-referencing (common average or linked ears)
- Normalize signals per channel
- **Data Splitting (Suggested Addition):** Implement Leave-One-Subject-Out (LOSO) cross-validation to prevent subject-level neural network overfitting.

### 3. Channel Subsampling
#### Experiment A — Channel Count Study (Controlled)
Fix input layouts to match typical clinical/consumer arrays:
- **5-channel set:** `{Fp1, Fp2, C3, C4, Oz}`
- **10-channel set:** add `{F3, F4, P3, P4, T7}`
- **15-channel set:** add remaining coverage points

#### Experiment B — Electrode Importance (🔥 Strongest Insight)
Determine strict locational relevance by removing electrodes sequentially. Ensure standard structural coverage (frontal, temporal, occipital).
$$ \text{Importance}(i) = \text{Error without electrode } i $$

---

## 🏗️ Phase 2: Baseline Models
### 4. Interpolation Methods
- Spherical spline interpolation
- Distance-weighted interpolation

### 5. Physics-Based Model
- Build BEM head model (brain, skull, scalp)
- Compute leadfield matrix
- Solve inverse problem (MNE/dSPM)
- Forward project resulting cortical estimation to the full electrode set

### 6. Pure AI Models
- Autoencoder (baseline spatio-compression)
- CNN (spatio-temporal dynamic model)
- *(Optional)* Graph Neural Network

---

## 🤖 Phase 3: Hybrid Model (Core Contribution)
### 7. Define Hybrid Framework
Integrate physics and neural predictions to synthesize a final estimation:
$$ \hat{X}_{final} = X_{physics} + f_\theta(X_{input}, X_{physics}) $$

### 8. Residual Learning Setup
Compute the missing "error residual" of the base physics model:
$$ R = X_{groundtruth} - X_{physics} $$
Train a neural network parameterized by $\theta$ to exclusively predict this residual $R$. 
- **Model Ablation (Suggested Addition):** Validate whether the neural network actually requires both $X_{input}$ and $X_{physics}$, or if predicting solely from $X_{physics}$ is sufficient.

### 9. (Optional) Frequency-Aware Learning
Divide prediction spaces by frequency domains:
- Low-frequency $\rightarrow$ Extracted via deterministic physics model
- High-frequency $\rightarrow$ Learned via AI model (train AI exclusively on high-frequency residuals)

---

## 📈 Phase 4: Evaluation
### 10. Signal-Level Performance Metric
- Root Mean Square Error (RMSE)
- Pearson Correlation Coefficient

### 11. Spatial Analysis (Suggested Addition)
- Topographical Spatial Correlation
- Peak Node Overlap (Does the reconstructed heatmap match the physical activation center?)

### 12. Frequency Analysis
- Power Spectral Density (PSD) comparison
- Band-wise evaluation errors:
  - Delta (0.5–4 Hz)
  - Theta (4–8 Hz)
  - Alpha (8–13 Hz)
  - Beta (13–30 Hz)
  - Gamma (>30 Hz)

### 13. Spatial Ablation Execution
Execute an electrode ablation sweep:
- Remove one electrode at a time globally
- Measure cumulative reconstruction degradation

### 14. Channel Scaling Study
Establish scaling efficiency equations by tracking performance across sets:
- $5 \rightarrow 19$
- $10 \rightarrow 19$
- $15 \rightarrow 19$
- **Deliverable Plot:** Error scale vs. Output spatial density (Number of Channels)

---

## 📊 Phase 5: Visualization
### 15. Generate Core Publication Plots
- Time-series layout plots (Overlaying Ground truth vs. Reconstructed arrays)
- Topographic spatial scalp heatmaps
- Power distributions displaying aggregated PSD comparisons 
- Electrode-level 19x19 or 64x64 error topology networks

---

## 🧾 Phase 6: Analysis & Reporting
### 16. Benchmark Cross-Comparison
- Spherical Interpolation vs. Analytical Physics vs. Pure AI vs. Deep Physics-AI Hybrid

### 17. Insights Generation Engine
- Empirically claim the minimum number of independent channels required to accurately reconstruct spatial brain activity
- Index the physical electrodes that define the greatest point of failure when missing
- Index the structural frequency bands (e.g., Alpha) that survive mathematical down-sampling the best
- Provide strict quantification backing the Hybrid model's theoretical advantages

### 18. Final Conclusions
- Explicitly define edge-cases where the deep mathematical reconstruction categorically fails
- Outline conditions where a 5-channel wearable device captures enough resolution equivalence to a clunky 19-channel clinical cap
- Summarize practical impacts for developing affordable wearable Brain-Computer Interfaces (BCI)
- **Latency Analysis (Suggested Addition):** Chart runtime latency metrics comparing MNE processing times vs. AI inference speeds to claim real-time operational limits.