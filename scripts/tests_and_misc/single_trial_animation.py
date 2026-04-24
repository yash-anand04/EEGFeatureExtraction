import mne
import pandas as pd
import os
import numpy as np
from mne.datasets import fetch_fsaverage
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path

# 1️⃣ LOAD EEG DATA
project_root = Path(__file__).resolve().parents[2]
eeg_csv_file = os.environ.get("EEG_CSV_FILE", "")
if not eeg_csv_file:
    raise ValueError("Set EEG_CSV_FILE to a trial eeg_data.csv path before running this script.")
df = pd.read_csv(eeg_csv_file)
data = df.filter(like='eeg').iloc[:, :19].values.T
ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
raw = mne.io.RawArray(data, mne.create_info(ch_names, 100, 'eeg'))
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage, on_missing='ignore')

# 2️⃣ BEM SETUP
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)
trans = mne.coreg.estimate_head_mri_t(subject='fsaverage', subjects_dir=subjects_dir)
print("✅ Transform ready!")

# 3️⃣ 🎥 SIMPLIFIED ANIMATION - DIRECT TOPOMAP
print("🎬 Creating electrode magnitude video...")

time = raw.times[:200]  # First 2 seconds
data_slice = raw.get_data()[:, :200]

fig, ax = plt.subplots(figsize=(12, 10), facecolor='black')
fig.patch.set_facecolor('black')

def animate(frame):
    ax.clear()
    ax.set_facecolor('black')
    
    # Get data at exact time point - shape (19,)
    current_data = data_slice[:, frame]
    
    # DIRECT TOPOMAP - NO EVOKED NEEDED!
    mne.viz.plot_topomap(
        current_data, 
        raw.info,
        axes=ax,
        show=False,
        vlim = (-200e-6, 200e-6),
        cmap='RdBu_r',
        size=3.0,
        outlines='head'
    )
    
    ax.set_title(f'Time: {time[frame]:.3f}s | Magnitude (µV)', color='white', fontsize=16)
    plt.tight_layout()

# Create & save animation
import os
gif_path = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'eeg_magnitude_animation.gif')
ani = FuncAnimation(fig, animate, frames=len(time), interval=50, blit=False, repeat=True)
ani.save(gif_path, writer='pillow', fps=20)
plt.close()

print(f"✅ Video saved: {gif_path}")

# 4️⃣ 3D BEM
mne.viz.plot_alignment(
    raw.info, trans=trans,
    subject='fsaverage',
    subjects_dir=subjects_dir,
    surfaces=['inner_skull', 'outer_skull', 'outer_skin'],
    coord_frame='head',
    show_axes=True
)
input("Press Enter to close 3D view...")

# 5️⃣ SAVE TRANSFORM
trans_path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'head_mri-trans.fif')
mne.write_trans(trans_path, trans, overwrite=True)
print("💾 Transform saved!")
