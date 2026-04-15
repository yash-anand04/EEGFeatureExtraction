import mne
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configuration
subjects_dir = r"C:\Users\unnat\mne_data\MNE-fsaverage-data"
data_base_path = r"C:\Users\unnat\Desktop\EEGFeatureExtraction\Subject_1\Baseline (with_audio_and_visual_stimulus)"
ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
n_trials = 20  # Start with 5 trials for analysis

class ChannelImportanceAnalyzer:
    def __init__(self, subjects_dir, ch_names):
        self.subjects_dir = subjects_dir
        self.ch_names = ch_names
        self.fwd = None
        self.src = None
        self.bem = None
        self._setup_forward_model()

    def _setup_forward_model(self):
        """Setup the forward model once"""
        print("Setting up forward model...")

        # Source space
        self.src = mne.setup_source_space(
            subject='fsaverage',
            subjects_dir=self.subjects_dir,
            spacing='oct6',
            add_dist=False
        )

        # BEM model with META-ANALYSIS conductivity values
        # Sources: Weighted averages from current meta-analysis
        # Brain (grey matter): 0.47 S/m, CSF: 1.71 S/m, Scalp: 0.41 S/m
        # Skull: 0.02 S/m (when modeled separately)
        model = mne.make_bem_model(
            subject='fsaverage',
            ico=4,
            subjects_dir=self.subjects_dir,
            conductivity=[0.47, 1.71, 0.41]  # Brain (GM), CSF, Scalp
        )
        self.bem = mne.make_bem_solution(model)

        # Load transformation
        trans = mne.read_trans(r'C:\Users\unnat\Desktop\EEGFeatureExtraction\head_mri-trans.fif')

        # Create info for forward model
        info = mne.create_info(self.ch_names, 100, 'eeg')
        montage = mne.channels.make_standard_montage('standard_1005')
        info.set_montage(montage, on_missing='ignore')

        # Forward solution
        self.fwd = mne.make_forward_solution(
            info, trans=trans,
            src=self.src, bem=self.bem,
            meg=False, eeg=True,
            mindist=5.0,
            n_jobs=1
        )
        print(f"Forward model ready: {self.fwd['nsource']} sources")

    def load_trial_data(self, trial_num, base_path):
        """Load data for a specific trial from a given base path"""
        trial_path = f"{base_path}\\trial_{trial_num:02d}\\eeg_data.csv"
        df = pd.read_csv(trial_path)
        data = df.filter(like='eeg').iloc[:, :19].values.T

        raw = mne.io.RawArray(data, mne.create_info(self.ch_names, 100, 'eeg'))
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='ignore')
        raw.set_eeg_reference(ref_channels='average', projection=True)

        return raw

    def load_all_trials(self, base_paths, n_trials):
        """Load all trials from multiple base paths"""
        raws = []
        for base_path in base_paths:
            for trial in range(1, n_trials + 1):
                try:
                    raw = self.load_trial_data(trial, base_path)
                    raws.append(raw)
                except Exception as e:
                    print(f"Warning: Could not load trial {trial} from {base_path}: {e}")
        return raws

    def compute_channel_importance(self, method='sensitivity'):
        """
        Compute channel importance using different methods:
        - 'sensitivity': Based on forward model sensitivity
        - 'variance': Based on channel variance across trials
        - 'reconstruction_error': Based on impact when channel is dropped
        """
        print(f"Computing channel importance using {method} method...")

        if method == 'sensitivity':
            return self._channel_sensitivity_importance()
        elif method == 'variance':
            return self._channel_variance_importance()
        elif method == 'reconstruction_error':
            return self._channel_reconstruction_importance()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _channel_sensitivity_importance(self):
        """Channel importance based on forward model sensitivity"""
        # Use the leadfield matrix to compute sensitivity
        leadfield = self.fwd['sol']['data']  # Shape: (n_channels, n_sources)

        # Compute RMS sensitivity for each channel across all sources
        sensitivities = np.sqrt(np.mean(leadfield**2, axis=1))

        # Normalize to 0-1 scale
        sensitivities = (sensitivities - np.min(sensitivities)) / (np.max(sensitivities) - np.min(sensitivities))

        importance_dict = dict(zip(self.ch_names, sensitivities))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def _channel_variance_importance(self, base_paths, n_trials):
        """Channel importance based on signal variance across all trials from all folders"""
        variances = []
        raws = self.load_all_trials(base_paths, n_trials)
        for raw in raws:
            raw.filter(1, 40)
            trial_variances = np.var(raw.get_data(), axis=1)
            variances.append(trial_variances)
        avg_variances = np.mean(variances, axis=0)
        avg_variances = (avg_variances - np.min(avg_variances)) / (np.max(avg_variances) - np.min(avg_variances))
        importance_dict = dict(zip(self.ch_names, avg_variances))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def _channel_reconstruction_importance(self):
        """Channel importance based on reconstruction error when dropped"""
        print("Computing reconstruction-based importance (this may take a while)...")

        full_stcs = []
        # First, compute full reconstructions for all trials
        for trial in range(1, n_trials + 1):
            raw = self.load_trial_data(trial)
            stc_full = self.compute_source_reconstruction(raw)
            full_stcs.append(stc_full.data)

        avg_full_stc = np.mean(full_stcs, axis=0)

        importance_scores = {}

        # For each channel, compute reconstruction error when dropped
        for ch_idx, ch_name in enumerate(self.ch_names):
            dropped_stcs = []

            for trial in range(1, n_trials + 1):
                raw = self.load_trial_data(trial)
                # Drop this channel
                raw_dropped = raw.copy().drop_channels([ch_name])
                stc_dropped = self.compute_source_reconstruction(raw_dropped)
                dropped_stcs.append(stc_dropped.data)

            avg_dropped_stc = np.mean(dropped_stcs, axis=0)

            # Compute MSE between full and dropped reconstructions
            mse = mean_squared_error(avg_full_stc.flatten(), avg_dropped_stc.flatten())
            importance_scores[ch_name] = mse  # Higher MSE = more important

        # Normalize (higher importance = higher score)
        max_score = max(importance_scores.values())
        importance_scores = {k: v/max_score for k, v in importance_scores.items()}

        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))

    def compute_source_reconstruction(self, raw):
        """Compute source reconstruction for given raw data"""
        raw.filter(1, 40)
        cov = mne.compute_raw_covariance(raw, tmin=0, tmax=None)

        inverse_operator = mne.minimum_norm.make_inverse_operator(
            raw.info, self.fwd, cov, loose=0.2, depth=0.8
        )

        evoked_data = raw.get_data()
        evoked = mne.EvokedArray(evoked_data, raw.info, tmin=0.0)

        stc = mne.minimum_norm.apply_inverse(
            evoked, inverse_operator,
            lambda2=1./9., method='dSPM'
        )

        return stc

    def analyze_channel_dropping(self, importance_ranking, n_channels_list=[19, 15, 10, 5]):
        """Analyze source reconstruction quality with different numbers of channels"""
        results = {}

        print("Analyzing channel dropping impact...")

        for n_ch in n_channels_list:
            print(f"\nTesting with {n_ch} channels...")

            # Select top N channels
            top_channels = list(importance_ranking.keys())[:n_ch]
            all_channels = list(importance_ranking.keys())
            dropped_channels = all_channels[n_ch:]

            print(f"  Keeping channels: {top_channels}")
            print(f"  Dropping channels: {dropped_channels}")

            trial_stcs = []
            for trial in range(1, n_trials + 1):
                raw = self.load_trial_data(trial)
                raw_subset = raw.copy().pick_channels(top_channels)
                stc = self.compute_source_reconstruction(raw_subset)
                trial_stcs.append(stc.data)

            # Compute metrics
            avg_stc = np.mean(trial_stcs, axis=0)
            std_stc = np.std(trial_stcs, axis=0)

            # Spatial consistency (lower std = more consistent)
            spatial_consistency = 1 / (1 + np.mean(std_stc))

            # Peak activation strength
            peak_activation = np.max(np.abs(avg_stc))

            # Active sources (above threshold)
            threshold = np.percentile(np.abs(avg_stc), 95)
            n_active_sources = np.sum(np.abs(avg_stc) > threshold)

            results[n_ch] = {
                'spatial_consistency': spatial_consistency,
                'peak_activation': peak_activation,
                'n_active_sources': n_active_sources,
                'channels_used': top_channels,
                'channels_dropped': dropped_channels
            }

        return results

def plot_importance_analysis(importance_ranking, dropping_results):
    """Plot the results of channel importance and dropping analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Channel importance ranking
    channels = list(importance_ranking.keys())
    scores = list(importance_ranking.values())

    ax1.bar(range(len(channels)), scores)
    ax1.set_xticks(range(len(channels)))
    ax1.set_xticklabels(channels, rotation=45)
    ax1.set_title('Channel Importance Ranking')
    ax1.set_ylabel('Importance Score')

    # Spatial consistency vs number of channels
    n_channels = list(dropping_results.keys())
    consistency = [dropping_results[n]['spatial_consistency'] for n in n_channels]

    ax2.plot(n_channels, consistency, 'o-', linewidth=2, markersize=8)
    ax2.set_title('Spatial Consistency vs Channel Count')
    ax2.set_xlabel('Number of Channels')
    ax2.set_ylabel('Spatial Consistency')
    ax2.grid(True)

    # Peak activation vs number of channels
    peak_acts = [dropping_results[n]['peak_activation'] for n in n_channels]

    ax3.plot(n_channels, peak_acts, 's-', linewidth=2, markersize=8, color='orange')
    ax3.set_title('Peak Activation vs Channel Count')
    ax3.set_xlabel('Number of Channels')
    ax3.set_ylabel('Peak Activation Strength')
    ax3.grid(True)

    # Active sources vs number of channels
    active_srcs = [dropping_results[n]['n_active_sources'] for n in n_channels]

    ax4.plot(n_channels, active_srcs, '^-', linewidth=2, markersize=8, color='green')
    ax4.set_title('Active Sources vs Channel Count')
    ax4.set_xlabel('Number of Channels')
    ax4.set_ylabel('Number of Active Sources')
    ax4.grid(True)

    plt.tight_layout()
    plt.show()

# Main analysis
if __name__ == "__main__":
    print("Starting Channel Importance Analysis...")

    # Suppress MNE verbose output
    mne.set_log_level('WARNING')

    analyzer = ChannelImportanceAnalyzer(subjects_dir, ch_names)

    # Define all three baseline folders
    base_paths = [
        r"C:\Users\unnat\Desktop\EEGFeatureExtraction\Subject_1\Baseline (in_silence)",
        r"C:\Users\unnat\Desktop\EEGFeatureExtraction\Subject_1\Baseline (with_audio_and_visual_stimulus)",
        r"C:\Users\unnat\Desktop\EEGFeatureExtraction\Subject_1\Baseline (with_music)"
    ]
    n_trials = 20

    print("\n1. Computing channel importance across ALL folders...")
    variance_importance = analyzer._channel_variance_importance(base_paths, n_trials)

    print("\nCombined Variance-based ranking (most to least important):")
    for i, (ch, score) in enumerate(variance_importance.items(), 1):
        print(f"{i:2d}. {ch:4s}  Score: {score:.3f}")

    # Output top channels for reconstruction
    top_15 = list(variance_importance.keys())[:15]
    top_10 = list(variance_importance.keys())[:10]
    top_5 = list(variance_importance.keys())[:5]

    print("\nTop 15 channels:", top_15)
    print("Top 10 channels:", top_10)
    print("Top 5 channels:", top_5)

    # Use top channels for dropping analysis
    print("\n2. Analyzing channel dropping impact with combined importance...")
    dropping_results = analyzer.analyze_channel_dropping(variance_importance, n_channels_list=[19, 15, 10, 5])

    print("\nChannel dropping results:")
    for n_ch, metrics in dropping_results.items():
        print(f"\n{n_ch} channels: Consistency={metrics['spatial_consistency']:.3f}, "
              f"Peak={metrics['peak_activation']:.2e}, Active Sources={metrics['n_active_sources']}")
        print(f"  Kept: {metrics['channels_used']}")
        print(f"  Dropped: {metrics['channels_dropped']}")

    # Plot results
    plot_importance_analysis(variance_importance, dropping_results)

    print("\nAnalysis complete! Check the plots for detailed insights.")