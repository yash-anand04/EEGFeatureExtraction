import os
import glob
import numpy as np
import pandas as pd
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne import EvokedArray

# Set matplotlib to background mode to prevent GUI popups and memory leaks
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# ==========================================
# 1. PATHS AND CONSTANTS
# ==========================================
BASE_DIR = r"C:\Users\unnat\Desktop\EEGFeatureExtraction\Subject_1"
SUBJECTS_DIR = r"C:\Users\unnat\mne_data\MNE-fsaverage-data"
TRANS_FILE = r"C:\Users\unnat\Desktop\EEGFeatureExtraction\head_mri-trans.fif"

ch_names = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
    'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 
    'Pz', 'P4', 'T6', 'O1', 'O2'
]

keep_channels = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Fp1', 'Fp2', 'O1', 'O2']
drop_channels = [ch for ch in ch_names if ch not in keep_channels]

# ==========================================
# 2. GLOBAL SETUP (Run Once)
# ==========================================
print("Building head model and forward solutions...")

info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

# 'oct5' for optimized ~2048 source space (Saves RAM/CPU)
src = mne.setup_source_space('fsaverage', spacing='oct5', subjects_dir=SUBJECTS_DIR)
model = mne.make_bem_model(subject='fsaverage', ico=4, 
                           conductivity=(0.33, 0.0132, 0.33), 
                           subjects_dir=SUBJECTS_DIR)
bem = mne.make_bem_solution(model)

# 1. Build the base free-orientation forward model
fwd_full_free = mne.make_forward_solution(info, trans=TRANS_FILE, src=src, bem=bem, 
                                          eeg=True, meg=False, mindist=5.0)

# 2. Convert to surface orientation (Keep FREE orientation for depth weighting calculations)
fwd_full_surf = mne.convert_forward_solution(fwd_full_free, surf_ori=True, 
                                             force_fixed=False, use_cps=True)

# 3. Create Sparse Forward for the inverse operator (Must be free orientation)
fwd_keep_surf = mne.pick_channels_forward(fwd_full_surf, include=keep_channels)

# 4. Create FIXED orientation forward JUST to extract the 2D G matrix for reconstruction
fwd_full_fixed = mne.convert_forward_solution(fwd_full_surf, force_fixed=True, use_cps=True)
G = fwd_full_fixed['sol']['data'] 

# Pre-compute forward indices for dropped channels to ensure exact spatial alignment
drop_idx_fwd = [fwd_full_fixed['info']['ch_names'].index(ch) for ch in drop_channels]

# ==========================================
# 3. BATCH PROCESSING LOOP
# ==========================================
print("Starting trial-level reconstruction and plotting...")
trial_files = glob.glob(os.path.join(BASE_DIR, 'Baseline (*)', 'trial_*', 'eeg_data.csv'))

results = []

# Realistic SNR for continuous, non-averaged EEG data
snr = 1.0  
lambda2 = 1.0 / snr**2

for file_path in trial_files:
    trial_dir = os.path.dirname(file_path)
    path_parts = file_path.split(os.sep)
    condition = path_parts[-3]
    trial_num = path_parts[-2]
    
    print(f"Processing: {condition} - {trial_num}")
    
    # Load data AND scale from Microvolts to Volts (* 1e-6)
    df = pd.read_csv(file_path)
    data = df.filter(like='eeg').iloc[:, :19].values.T * 1e-6
    
    raw_full = mne.io.RawArray(data, info, verbose=False)
    raw_full.set_montage(montage)
    raw_full.set_eeg_reference('average', projection=True, verbose=False)
    raw_full.apply_proj()
    raw_full.filter(l_freq=1.0, h_freq=40.0, verbose=False)
    
    # Subset to kept channels for inverse operator
    raw_keep = raw_full.copy().pick(keep_channels)
    cov = mne.compute_raw_covariance(raw_keep, tmin=0, tmax=None, verbose=False)
    
    # Compute Inverse (loose=0.0 locks the 3D surface model down to 1D fixed for the math)
    inv = make_inverse_operator(raw_keep.info, fwd_keep_surf, cov, loose=0.0, depth=0.8, verbose=False)
    
    # Treat continuous baseline as Evoked to apply weights dynamically
    evoked_keep = EvokedArray(raw_keep.get_data(), raw_keep.info, tmin=0.0, verbose=False)
    
    # Apply Inverse (method='MNE' keeps output in physical units rather than statistical Z-scores)
    stc = apply_inverse(evoked_keep, inv, lambda2=lambda2, method='MNE', pick_ori=None, verbose=False)
    
    # ------------------------------------------
    # FAST MATH RECONSTRUCTION
    # ------------------------------------------
    recon_evoked = mne.apply_forward(fwd_full_fixed, stc, raw_full.info, verbose=False)
    recon_all = recon_evoked.data
    
    # Use MNE `picks` to slice cleanly, and our precomputed indices for the reconstructed matrix
    y_true_mat = raw_full.get_data(picks=drop_channels)
    y_recon_mat = recon_all[drop_idx_fwd, :]
    
    # ------------------------------------------
    # VECTORIZED METRICS
    # ------------------------------------------
    mu_true = np.mean(y_true_mat, axis=1, keepdims=True)
    mu_recon = np.mean(y_recon_mat, axis=1, keepdims=True)
    
    y_true_c = y_true_mat - mu_true
    y_recon_c = y_recon_mat - mu_recon
    
    num = np.sum(y_true_c * y_recon_c, axis=1)
    den = np.sqrt(np.sum(y_true_c**2, axis=1) * np.sum(y_recon_c**2, axis=1))
    r_vals = num / den
    
    mse_vals = np.mean((y_true_mat - y_recon_mat)**2, axis=1)
    
    # ------------------------------------------
    # PLOTTING & SAVING
    # ------------------------------------------
    times = np.arange(y_true_mat.shape[1]) / 100.0 
    plots_dir = os.path.join(trial_dir, "Reconstruction_Plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for i, ch in enumerate(drop_channels):
        # Save Metrics
        results.append({
            'Condition': condition,
            'Trial': trial_num,
            'Channel': ch,
            'Pearson_r': r_vals[i],
            'MSE': mse_vals[i]
        })
        
        # Plot Original vs Reconstructed (Multiply by 1e6 to bring back to µV for readable axes)
        fig, ax = plt.subplots(figsize=(12, 4))
        
        ax.plot(times, y_true_mat[i] * 1e6, label='Original (Ground Truth)', 
                color='black', alpha=0.6, linewidth=1.2)
        ax.plot(times, y_recon_mat[i] * 1e6, label='Reconstructed', 
                color='red', alpha=0.8, linestyle='--', linewidth=1.2)
        
        ax.set_title(f"Channel {ch} | r = {r_vals[i]:.3f} | MSE = {mse_vals[i]:.3e}")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude (µV)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
        # Save locally and explicitly close the figure to free up RAM
        plot_path = os.path.join(plots_dir, f"{ch}_recon.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

# ==========================================
# 4. AGGREGATION
# ==========================================
df_results = pd.DataFrame(results)
print("\nBatch Complete! Visuals saved in respective trial folders.")

# Save master metrics summary to the root Subject folder
summary_path = os.path.join(BASE_DIR, "all_reconstruction_metrics.csv")
df_results.to_csv(summary_path, index=False)
print(f"Metrics saved to: {summary_path}")