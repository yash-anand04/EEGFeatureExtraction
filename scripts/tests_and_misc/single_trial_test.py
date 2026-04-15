import mne
import pandas as pd
import os
from mne.datasets import fetch_fsaverage
import matplotlib.pyplot as plt

# 1️⃣ LOAD EEG DATA
df = pd.read_csv(r"C:\Users\unnat\Desktop\EEGFeatureExtraction\Subject_1\Baseline (in_silence)\trial_02\eeg_data.csv")
data = df.filter(like='eeg').iloc[:, :19].values.T
ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
raw = mne.io.RawArray(data, mne.create_info(ch_names, 100, 'eeg'))
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage, on_missing='ignore')

# 2️⃣ CORRECT subjects_dir = PARENT of MNE-fsaverage-data
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)  # C:\Users\unnat\mne_data
print(f"✅ subjects_dir: {subjects_dir}")
print(f"✅ fsaverage lives here: {fs_dir}")

# 3️⃣ CHECK BEM EXISTS (should be TRUE)
bem_path = os.path.join(fs_dir, 'bem', 'inner_skull.surf')
print(f"✅ BEM exists: {os.path.exists(bem_path)}")

# 4️⃣ TRANSFORM + BEM PLOT
trans = mne.coreg.estimate_head_mri_t(subject='fsaverage', subjects_dir=subjects_dir)
print("🔍 TRANSFORM MATRIX:")
print(trans)

# FULL FINAL CODE - Window stays open:
mne.viz.plot_alignment(
    raw.info, trans=trans,
    subject='fsaverage',
    subjects_dir=subjects_dir,
    surfaces=['inner_skull', 'outer_skull', 'outer_skin'],
    coord_frame='head',
    show_axes=True
)
input("Press Enter to close 3D view and continue...")  # Manual control


mne.write_trans('head_mri-trans.fif', trans, overwrite=True)
