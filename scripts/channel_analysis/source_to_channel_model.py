import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os
import pandas as pd
import mne
from pathlib import Path

# Dummy loader: replace with your actual data loading
# original_data: (samples, channels)
# source_activations: (samples, n_sources)
def load_data():
    import os
    import pandas as pd
    import mne
    project_root = Path(__file__).resolve().parents[2]
    subjects_dir = os.environ.get("MNE_SUBJECTS_DIR", str(project_root / "mne_data" / "MNE-fsaverage-data"))
    trans_path = os.environ.get("EEG_TRANS_FILE", str(project_root / "head_mri-trans.fif"))
    ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
    conditions = [
        "Baseline (in_silence)",
        "Baseline (with_audio_and_visual_stimulus)",
        "Baseline (with_music)"
    ]
    base_dir = os.environ.get("EEG_BASE_DIR", "")
    if not base_dir:
        raise ValueError("Set EEG_BASE_DIR to the folder containing Baseline (*) trial directories.")
    trial_count = 20
    channel_data_list = []
    source_data_list = []
    for cond in conditions:
        cond_dir = os.path.join(base_dir, cond)
        for i in range(1, trial_count+1):
            ch_csv = os.path.join(cond_dir, f'trial_{i:02d}', 'eeg_data.csv')
            if os.path.exists(ch_csv):
                ch_df = pd.read_csv(ch_csv)
                # Select only numeric EEG columns (first 19 columns or those named 'Fp1', ...)
                if set(ch_names).issubset(set(ch_df.columns)):
                    eeg_data = ch_df[ch_names].values.T  # shape (channels, samples)
                else:
                    eeg_data = ch_df.select_dtypes(include=[float, int]).iloc[:, :19].values.T
                channel_data_list.append(eeg_data.T)  # shape (samples, channels)
                # Compute sources using MNE
                info = mne.create_info(ch_names, 100, 'eeg')
                raw = mne.io.RawArray(eeg_data, info)
                montage = mne.channels.make_standard_montage('standard_1005')
                raw.set_montage(montage, on_missing='ignore')
                raw.set_eeg_reference(ref_channels='average', projection=True)
                raw.filter(1, 40)
                cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)
                src = mne.setup_source_space(
                    subject='fsaverage',
                    subjects_dir=subjects_dir,
                    spacing='oct6',
                    add_dist=False
                )
                model = mne.make_bem_model(
                    subject='fsaverage',
                    ico=4,
                    subjects_dir=subjects_dir,
                    conductivity=[0.47, 1.71, 0.41]
                )
                bem = mne.make_bem_solution(model)
                trans = mne.read_trans(trans_path)
                fwd = mne.make_forward_solution(
                    info, trans=trans,
                    src=src, bem=bem,
                    meg=False, eeg=True,
                    mindist=5.0,
                    n_jobs=1
                )
                inverse_operator = mne.minimum_norm.make_inverse_operator(
                    raw.info, fwd, cov, loose=0.2, depth=0.8
                )
                evoked_data = raw.get_data()
                evoked = mne.EvokedArray(evoked_data, raw.info, tmin=0.0)
                stc = mne.minimum_norm.apply_inverse(
                    evoked, inverse_operator,
                    lambda2=1./9., method='dSPM'
                )
                # Use stc.data.T (samples, n_sources)
                source_data_list.append(stc.data.T)
    original_data = np.concatenate(channel_data_list, axis=0)  # (samples, channels)
    source_activations = np.concatenate(source_data_list, axis=0)  # (samples, n_sources)
    return original_data, source_activations

class SourceToChannelNet(nn.Module):
    def __init__(self, n_sources, n_channels):
        super().__init__()
        self.fc1 = nn.Linear(n_sources, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_and_evaluate(X, y, n_channels, use_sources=True):
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    model = SourceToChannelNet(X.shape[1], n_channels)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = loss_fn(out, y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).numpy()
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    r2 = r2_score(y, y_pred_inv)
    mse = mean_squared_error(y, y_pred_inv)
    print(f"R2 score: {r2:.4f}, MSE: {mse:.4f}")
    return r2, mse

def main():
    original_data, source_activations = load_data()
    n_channels = original_data.shape[1]
    channel_counts = [5, 10, 15]
    for n_in in channel_counts:
        print(f"\n=== {n_in} Input Channels ===")
        # Model 1: Using only channel subset
        channel_subset = original_data[:, :n_in]
        print("Model with channel subset:")
        r2_ch, mse_ch = train_and_evaluate(channel_subset, original_data, n_channels, use_sources=False)
        # Model 2: Using source activations
        print("Model with source activations:")
        r2_src, mse_src = train_and_evaluate(source_activations, original_data, n_channels, use_sources=True)
        print(f"Channel features R2: {r2_ch:.4f}, Source features R2: {r2_src:.4f}")

if __name__ == "__main__":
    main()
