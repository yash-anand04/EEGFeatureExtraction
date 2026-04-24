import os
import numpy as np
import pandas as pd
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne import EvokedArray
import matplotlib.pyplot as plt
from pathlib import Path

# --- 1. SET YOUR PATH TO ONE TRIAL ---
project_root = Path(__file__).resolve().parents[2]
file_path = os.environ.get("EEG_CSV_FILE", "")
if not file_path:
	raise ValueError("Set EEG_CSV_FILE to a trial eeg_data.csv path before running this script.")
TRANS_FILE = os.environ.get("EEG_TRANS_FILE", str(project_root / "head_mri-trans.fif"))
SUBJECTS_DIR = os.environ.get("MNE_SUBJECTS_DIR", str(project_root / "mne_data" / "MNE-fsaverage-data"))

ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
keep_channels = ['Fz', 'Cz', 'Pz', 'C3', 'C4', 'P3', 'P4', 'Fp1', 'Fp2', 'O1', 'O2']

# --- 2. BUILD FORWARD/INVERSE ---
info = mne.create_info(ch_names=ch_names, sfreq=100.0, ch_types='eeg')
montage = mne.channels.make_standard_montage('standard_1020')
info.set_montage(montage)

src = mne.setup_source_space('fsaverage', spacing='oct5', subjects_dir=SUBJECTS_DIR)
model = mne.make_bem_model(subject='fsaverage', ico=4, conductivity=(0.33, 0.0132, 0.33), subjects_dir=SUBJECTS_DIR)
bem = mne.make_bem_solution(model)

fwd_free = mne.make_forward_solution(info, trans=TRANS_FILE, src=src, bem=bem, eeg=True, meg=False)
fwd_surf = mne.convert_forward_solution(fwd_free, surf_ori=True, force_fixed=False, use_cps=True)
fwd_keep_surf = mne.pick_channels_forward(fwd_surf, include=keep_channels)
fwd_fixed = mne.convert_forward_solution(fwd_surf, force_fixed=True, use_cps=True)

# --- 3. PROCESS TRIAL ---
df = pd.read_csv(file_path)
data = df.filter(like='eeg').iloc[:, :19].values.T * 1e-6

raw_full = mne.io.RawArray(data, info, verbose=False)
raw_full.set_montage(montage)
raw_full.set_eeg_reference('average', projection=True, verbose=False)
raw_full.apply_proj()
raw_full.filter(l_freq=1.0, h_freq=40.0, verbose=False)

raw_keep = raw_full.copy().pick(keep_channels)
cov = mne.compute_raw_covariance(raw_keep, tmin=0, tmax=None, verbose=False)

inv = make_inverse_operator(raw_keep.info, fwd_keep_surf, cov, loose=0.0, depth=0.8, verbose=False)
evoked_keep = EvokedArray(raw_keep.get_data(), raw_keep.info, tmin=0.0, verbose=False)

# MNE solver for physical units
stc = apply_inverse(evoked_keep, inv, lambda2=1.0, method='MNE', pick_ori=None, verbose=False)

# NEW RECONSTRUCTION METHOD
recon_evoked = mne.apply_forward(fwd_fixed, stc, raw_full.info, verbose=False)

# --- 4. SANITY CHECK ON A "KEPT" CHANNEL ---
test_ch = 'Cz' # This is a KEPT channel!
idx_orig = raw_full.ch_names.index(test_ch)
idx_recon = recon_evoked.ch_names.index(test_ch)

y_true = raw_full.get_data()[idx_orig]
y_recon = recon_evoked.data[idx_recon]

r = np.corrcoef(y_true, y_recon)[0, 1]

plt.figure(figsize=(10, 4))
plt.plot(y_true * 1e6, label='Original', color='black')
plt.plot(y_recon * 1e6, label='Reconstructed', color='red', linestyle='--')
plt.title(f"SANITY CHECK: Kept Channel {test_ch} | r = {r:.3f}")
plt.legend()
plt.show()