import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
# YOUR DATA + CORRECT PATHS
subjects_dir = r"C:\Users\unnat\mne_data\MNE-fsaverage-data"  # PARENT directory
fs_dir = r"C:\Users\unnat\mne_data\MNE-fsaverage-data\fsaverage"  # ACTUAL fsaverage

# Load your EEG data (same as before)
df = pd.read_csv(r"C:\Users\unnat\Desktop\EEGFeatureExtraction\Subject_1\Baseline (with_audio_and_visual_stimulus)\trial_08\eeg_data.csv")
data = df.filter(like='eeg').iloc[:, :19].values.T
ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
raw = mne.io.RawArray(data, mne.create_info(ch_names, 100, 'eeg'))
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage, on_missing='ignore')
raw.set_eeg_reference(ref_channels='average', projection=True)

trans = mne.read_trans('head_mri-trans.fif')

# ✅ FIXED subjects_dir
print(f"Looking for surf in: {os.path.join(subjects_dir, 'MNE-fsaverage-data', 'fsaverage', 'surf', 'lh.white')}")

# 1️⃣ SOURCE SPACE
src = mne.setup_source_space(
    subject='fsaverage',  # ← Use actual folder name
    subjects_dir=subjects_dir,     # ← Parent directory
    spacing='oct6',
    add_dist=False
)

# 2️⃣ BEM MODEL
model = mne.make_bem_model(
    subject='fsaverage',
    ico=4,
    subjects_dir=subjects_dir,
    conductivity=[0.3, 0.01, 0.3]
)
bem = mne.make_bem_solution(model)

# 3️⃣ FORWARD SOLUTION
fwd = mne.make_forward_solution(
    raw.info, trans=trans,
    src=src, bem=bem,
    meg=False, eeg=True,
    mindist=5.0,
    n_jobs=1
)

print(f"✅ Forward solution: {fwd['nsource']} sources")

# 4️⃣ COVARIANCE + INVERSE
raw.filter(1, 40)
cov = mne.compute_raw_covariance(raw, tmin=0, tmax=10)

# FILTER + COVARIANCE (✅ WORKING)
raw.filter(1, 40)
cov = mne.compute_raw_covariance(raw, tmin=0, tmax=10)

# INVERSE OPERATOR (✅ WORKING)
inverse_operator = mne.minimum_norm.make_inverse_operator(
    raw.info, fwd, cov, loose=0.2, depth=0.8
)

# ✅ FIXED: Create Evoked manually from Raw data
evoked_data = raw.get_data()  # Shape: (19, n_times)
evoked = mne.EvokedArray(evoked_data, raw.info, tmin=0.0)  # 2D only!

print(f"✅ Evoked created: {evoked.data.shape}")

# SOURCE ESTIMATE (your EEG → brain sources!)
stc = mne.minimum_norm.apply_inverse(
    evoked, inverse_operator, 
    lambda2=1./9., method='dSPM'
)

# 6️⃣ 🧠 VISUALIZE BRAIN SOURCES
stc.plot(
    subjects_dir=subjects_dir,
    subject='fsaverage',
    hemi='both', 
    time_viewer=True,
    initial_time=0.1,
    size=(1200, 800)
)
input("Press Enter to close brain visualization...")  # ← KEEPS WINDOW OPEN

# # 7️⃣ GLASS BRAIN VISUALIZATION (shows sources inside brain)
# from mne.viz import plot_glass_brain
# glass_brain_fig = plot_glass_brain(stc, display_mode='ortho', plot_abs=False, cmap='hot', threshold=0.5)
# plt.show()

# # 8️⃣ VOLUME SOURCE ESTIMATE (if available)
# if hasattr(stc, 'as_volume'):  # Only works for volume source estimates
#     try:
#         vol_stc = stc.as_volume(
#             src, mri_resolution=True, mri_space=True
#         )
#         vol_stc.plot(
#             subject='fsaverage',
#             subjects_dir=subjects_dir,
#             initial_time=0.1,
#             opacity=0.5,
#             size=(1200, 800)
#         )
#     except Exception as e:
#         print(f"Volume rendering not available: {e}")

