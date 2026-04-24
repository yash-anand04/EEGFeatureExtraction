import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
import mne
# hyperparameter tuning
EPOCHS = 100 # idk we will have to tune with optuna later
BATCH_SIZE = 8 # increase based on gpu
LEARNING_RATE = 1e-3


#folders and paths
project_root = Path(__file__).resolve().parents[2]
BASE_DIR = os.environ.get("EEG_BASE_DIR", "")
if not BASE_DIR:
    raise ValueError("Set EEG_BASE_DIR to the folder containing Baseline (*) trial directories.")
SUBJECTS_DIR = os.environ.get("MNE_SUBJECTS_DIR", str(project_root / "mne_data" / "MNE-fsaverage-data"))
TRANS_FILE = os.environ.get("EEG_TRANS_FILE", str(project_root / "head_mri-trans.fif"))
CONDUCTIVITIES = []

TARGET_CH = "T6" #could by any channel, this is just for first run testing


ch_names = [
 'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
    'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 
    'Pz', 'P4', 'T6', 'O1', 'O2'
]

keep_channels = [] # will fill this later
drop_channels = [ch for ch in ch_names if ch not in keep_channels]

# mne setup
print("building physics model")
info = mne.create_info(ch_names=ch_names, sfreq=100, ch_types="eeg")
montage = mne.channels.make_standard_montage("standard_1020")
info.set_montage(montage)

src = mne.setup_source_space('fsaverage', spacing='oct5', subjects_dir=SUBJECTS_DIR)

model = mne.make_bem_model(subject='fsaverage', ico=4, conductivity=CONDUCTIVITIES,
                           subjects_dir=SUBJECTS_DIR)
bem = mne.make_bem_solution(model)

fwd_free = mne.make_forward_solution(info, trans=TRANS_FILE, src=src, bem=bem, eeg=True, meg=False)
fwd_surf = mne.convert_forward_solution(fwd = fwd_free, surf_ori=True)
fwd_keep_surf = mne.pick_channels_forward(fwd_surf, include=keep_channels)
fwd_fixed = mne.convert_forward_solution(fwd_surf, force_fixed=True)





