from sklearn.metrics import r2_score

import os
import glob
import numpy as np
import pandas as pd
import mne
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from mne.minimum_norm import make_inverse_operator, apply_inverse
# Set matplotlib to background mode if running headless, or inline if in Jupyter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
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
# 1. PATHS, CONSTANTS & CONFIG
# ==========================================
project_root = Path(__file__).resolve().parents[2]
BASE_DIR = os.environ.get("EEG_BASE_DIR", "")
if not BASE_DIR:
    raise ValueError("Set EEG_BASE_DIR to the folder containing Baseline (*) trial directories.")
SUBJECTS_DIR = os.environ.get("MNE_SUBJECTS_DIR", str(project_root / "mne_data" / "MNE-fsaverage-data"))
TRANS_FILE = os.environ.get("EEG_TRANS_FILE", str(project_root / "head_mri-trans.fif"))

ch_names = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
    'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 
    'Pz', 'P4', 'T6', 'O1', 'O2'
]

keep_channels = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Fp1', 'Fp2', 'O1', 'O2']
drop_channels = [ch for ch in ch_names if ch not in keep_channels]

# The specific channel we want the AI to fix in this run
TARGET_CH = 'T6'

# Deep Learning Hyperparameters
EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.001

# ==========================================
# 2. GLOBAL MNE SETUP (Physics Base)
# ==========================================
print("--- STAGE 1: Building Physics Model ---")
info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

src = mne.setup_source_space('fsaverage', spacing='oct5', subjects_dir=SUBJECTS_DIR)
# Modern 1:25 conductivity ratio for better skull penetration
model = mne.make_bem_model(subject='fsaverage', ico=4, conductivity=(0.33, 0.0132, 0.33), subjects_dir=SUBJECTS_DIR)
bem = mne.make_bem_solution(model)

fwd_free = mne.make_forward_solution(info, trans=TRANS_FILE, src=src, bem=bem, eeg=True, meg=False)
fwd_surf = mne.convert_forward_solution(fwd_free, surf_ori=True, force_fixed=False, use_cps=True)
fwd_keep_surf = mne.pick_channels_forward(fwd_surf, include=keep_channels)
fwd_fixed = mne.convert_forward_solution(fwd_surf, force_fixed=True, use_cps=True)

# ==========================================
# 3. DATASET GENERATION
# ==========================================
print("\n--- STAGE 2: Processing Trials for PyTorch Dataset ---")
trial_files = glob.glob(os.path.join(BASE_DIR, 'Baseline (*)', 'trial_*', 'eeg_data.csv'))

X_list = []
Y_list = []

snr = 1.0  
lambda2 = 1.0 / snr**2

for file_path in trial_files:
    # Load and scale to Volts
    df = pd.read_csv(file_path)
    data = df.filter(like='eeg').iloc[:, :19].values.T * 1e-6
    
    raw_full = mne.io.RawArray(data, info, verbose=False)
    raw_full.set_montage(montage)
    raw_full.set_eeg_reference('average', projection=True, verbose=False)
    raw_full.apply_proj()
    raw_full.filter(l_freq=1.0, h_freq=40.0, verbose=False)
    
    # Physics Reconstruction
    raw_keep = raw_full.copy().pick(keep_channels)
    cov = mne.compute_raw_covariance(raw_keep, tmin=0, tmax=None, verbose=False)
    inv = make_inverse_operator(raw_keep.info, fwd_keep_surf, cov, loose=0.0, depth=0.8, verbose=False)
    evoked_keep = mne.EvokedArray(raw_keep.get_data(), raw_keep.info, tmin=0.0, verbose=False)
    stc = apply_inverse(evoked_keep, inv, lambda2=lambda2, method='MNE', verbose=False)
    
    # Re-project with MNE to preserve average reference polarity
    recon_evoked = mne.apply_forward(fwd_fixed, stc, raw_full.info, verbose=False)
    
    # --- Format Tensors for Deep Learning ---
    # Convert data back to Microvolts (µV) for stable neural network gradients
    
    # 1. Get the 11 Kept Channels (Shape: 11 x 3000)
    kept_data = raw_keep.get_data() * 1e6 
    
    # 2. Get MNE's blurry guess for the Target Channel (Shape: 1 x 3000)
    idx_recon = recon_evoked.ch_names.index(TARGET_CH)
    mne_recon_target = recon_evoked.data[idx_recon : idx_recon+1, :] * 1e6
    
    # 3. Build Network Input: Stack kept channels + MNE guess (Shape: 12 x 3000)
    X_input = np.vstack([kept_data, mne_recon_target])
    
    # 4. Get the true Target Channel (Shape: 1 x 3000)
    idx_true = raw_full.ch_names.index(TARGET_CH)
    y_true_target = raw_full.get_data()[idx_true : idx_true+1, :] * 1e6
    
    # 5. Calculate the true residual (The spikes MNE missed)
    y_residual = y_true_target - mne_recon_target
    
    X_list.append(X_input)
    Y_list.append(y_residual)

print(f"Extracted {len(X_list)} trials for training.")

# Convert to PyTorch Tensors
X_tensor = torch.FloatTensor(np.array(X_list))
Y_tensor = torch.FloatTensor(np.array(Y_list))
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 4. DEEP LEARNING MODEL DEFINITION
# ==========================================


# --- Simple 1D CNN (Original) ---
class EEGResidualNetSimple(nn.Module):
    def __init__(self, in_channels=12):
        super(EEGResidualNetSimple, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, padding='same'),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 16, kernel_size=7, padding='same'),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 8, kernel_size=3, padding='same'),
            nn.GELU(),
            nn.Conv1d(8, 1, kernel_size=1)
        )
    def forward(self, x):
        return self.net(x)



# --- 1D ResNet Block ---
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample or (in_channels != out_channels)
        if self.downsample:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
    def forward(self, x):
        identity = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.gelu(out)
        return out

# --- 1D ResNet for EEG Residuals (New) ---
class EEGResidualNet(nn.Module):
    def __init__(self, in_channels=12):
        super().__init__()
        self.initial = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.layer1 = ResBlock1D(32, 32, kernel_size=7)
        self.layer2 = ResBlock1D(32, 32, kernel_size=7)
        self.layer3 = ResBlock1D(32, 16, kernel_size=5, downsample=True)
        self.layer4 = ResBlock1D(16, 8, kernel_size=3, downsample=True)
        self.final = nn.Conv1d(8, 1, kernel_size=1)
    def forward(self, x):
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final(x)
        return x

# ==========================================
# 5. TRAINING LOOP
# ==========================================

def train_and_evaluate_model(model_class, model_name, X_tensor, Y_tensor, dataloader, keep_channels, BASE_DIR, TARGET_CH):
    print(f"\n--- Training {model_name} for {TARGET_CH} ---")
    model = model_class(in_channels=len(keep_channels) + 1)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for batch_X, batch_Y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_Y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"{model_name} | Epoch {epoch+1}/{EPOCHS} | L1 Loss (MAE in µV): {epoch_loss/len(dataloader):.4f}")
    # Evaluation
    print(f"\n--- {model_name} Final Evaluation Plot ---")
    model.eval()
    with torch.no_grad():
        test_X = X_tensor[0:1]
        true_residual = Y_tensor[0:1].numpy().squeeze()
        predicted_residual = model(test_X).numpy().squeeze()
    mne_base = test_X[0, -1, :].numpy()
    true_signal = mne_base + true_residual
    ai_reconstruction = mne_base + predicted_residual
    times = np.arange(3000) / 100.0
    plt.figure(figsize=(14, 6))
    plt.plot(times, true_signal, label='Original (Ground Truth)', color='black', alpha=0.7, linewidth=1.5)
    plt.plot(times, mne_base, label='Physics Only (MNE, Smoothed)', color='red', linestyle='--', alpha=0.8, linewidth=1.5)
    plt.plot(times, ai_reconstruction, label=f'Hybrid ({model_name})', color='blue', alpha=0.8, linewidth=1.5)
    plt.title(f"Hybrid AI Reconstruction for Dropped Channel: {TARGET_CH} ({model_name})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (µV)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    save_img_path = os.path.join(BASE_DIR, f"Hybrid_Recon_{TARGET_CH}_{model_name}.png")
    plt.savefig(save_img_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nDone! Check your folder for: {save_img_path}")
    # Optionally, return model and results for further analysis
    return model, ai_reconstruction, true_signal, mne_base


# Train and evaluate both models
model_simple, ai_recon_simple, true_signal, mne_base = train_and_evaluate_model(
    EEGResidualNetSimple, "SimpleCNN", X_tensor, Y_tensor, dataloader, keep_channels, BASE_DIR, TARGET_CH)

model_resnet, ai_recon_resnet, _, _ = train_and_evaluate_model(
    EEGResidualNet, "ResNet", X_tensor, Y_tensor, dataloader, keep_channels, BASE_DIR, TARGET_CH)


def leave_one_trial_out_cv(model_class, model_name, X_tensor, Y_tensor, keep_channels, BASE_DIR, TARGET_CH):
    n_trials = X_tensor.shape[0]
    r2_scores = []
    peak_overlaps = []
    print(f"\n===== {model_name} Leave-One-Trial-Out CV for {TARGET_CH} =====")
    from scipy.signal import find_peaks
    n_peaks = 100
    for i in range(n_trials):
        idx = list(range(n_trials))
        idx_train = idx[:i] + idx[i+1:]
        idx_test = [i]
        X_train = X_tensor[idx_train]
        Y_train = Y_tensor[idx_train]
        X_test = X_tensor[idx_test]
        Y_test = Y_tensor[idx_test]
        train_dataset = TensorDataset(X_train, Y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        model = model_class(in_channels=len(keep_channels) + 1)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.L1Loss()
        model.train()
        for epoch in range(EPOCHS):
            for batch_X, batch_Y in train_loader:
                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_Y)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_residual = model(X_test).cpu().numpy().squeeze()
        mne_base = X_test[0, -1, :].cpu().numpy()
        true_residual = Y_test[0].cpu().numpy().squeeze()
        true_signal = mne_base + true_residual
        ai_recon = mne_base + pred_residual
        true_signal = np.ravel(true_signal)
        ai_recon = np.ravel(ai_recon)
        r2 = r2_score(true_signal, ai_recon)
        r2_scores.append(r2)
        # --- Top 100 peak overlap ---
        peaks_true, _ = find_peaks(true_signal)
        peaks_recon, _ = find_peaks(ai_recon)
        if len(peaks_true) > 0:
            top_true = peaks_true[np.argsort(np.abs(true_signal[peaks_true]))[-n_peaks:]] if len(peaks_true) >= n_peaks else peaks_true
        else:
            top_true = np.array([], dtype=int)
        if len(peaks_recon) > 0:
            top_recon = peaks_recon[np.argsort(np.abs(ai_recon[peaks_recon]))[-n_peaks:]] if len(peaks_recon) >= n_peaks else peaks_recon
        else:
            top_recon = np.array([], dtype=int)
        match_count = 0
        for pt in top_true:
            if np.any(np.abs(top_recon - pt) <= 2):
                match_count += 1
        peak_overlap = match_count / max(len(top_true), 1)
        peak_overlaps.append(peak_overlap)
        print(f"Trial {i+1}/{n_trials} | R2: {r2:.4f} | Peak Overlap: {peak_overlap:.2f}")
        if i == 0:
            times = np.arange(mne_base.shape[-1]) / 100.0
            plt.figure(figsize=(14, 6))
            plt.plot(times, true_signal, label='Original (Ground Truth)', color='black', alpha=0.7, linewidth=1.5)
            plt.plot(times, mne_base, label='Physics Only (MNE, Smoothed)', color='red', linestyle='--', alpha=0.8, linewidth=1.5)
            plt.plot(times, ai_recon, label=f'Hybrid ({model_name})', color='blue', alpha=0.8, linewidth=1.5)
            plt.title(f"Hybrid AI Reconstruction for Dropped Channel: {TARGET_CH} ({model_name}, Trial {i+1})")
            plt.xlabel("Time (seconds)")
            plt.ylabel("Amplitude (µV)")
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            save_img_path = os.path.join(BASE_DIR, f"Hybrid_Recon_{TARGET_CH}_{model_name}_Trial{i+1}.png")
            plt.savefig(save_img_path, dpi=200, bbox_inches='tight')
            plt.close()
    r2_scores = np.array(r2_scores)
    peak_overlaps = np.array(peak_overlaps)
    print(f"\n{model_name} Leave-One-Trial-Out CV Results:")
    print(f"Mean R2: {r2_scores.mean():.4f} | Std R2: {r2_scores.std():.4f}")
    print(f"Mean Peak Overlap: {peak_overlaps.mean():.4f} | Std Peak Overlap: {peak_overlaps.std():.4f}")
    return r2_scores, peak_overlaps

# Run leave-one-trial-out CV for both models
r2_simple = leave_one_trial_out_cv(
    EEGResidualNetSimple, "SimpleCNN", X_tensor, Y_tensor, keep_channels, BASE_DIR, TARGET_CH)

r2_resnet = leave_one_trial_out_cv(
    EEGResidualNet, "ResNet", X_tensor, Y_tensor, keep_channels, BASE_DIR, TARGET_CH)