import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import warnings
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
warnings.filterwarnings('ignore')

# Configuration
subjects_dir = r"C:\Users\unnat\mne_data\MNE-fsaverage-data"
# Use all three baseline conditions from Subject_1
data_conditions = [
    r"C:\Users\unnat\Desktop\EEGFeatureExtraction\Subject_1\Baseline (in_silence)",
    r"C:\Users\unnat\Desktop\EEGFeatureExtraction\Subject_1\Baseline (with_audio_and_visual_stimulus)",
    r"C:\Users\unnat\Desktop\EEGFeatureExtraction\Subject_1\Baseline (with_music)"
]
ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
n_trials_per_condition = 20  # 20 trials per condition
total_trials = 60  # 3 conditions × 20 trials each

class FastChannelReconstructor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ChannelReconstructionAnalyzer:
    def get_model_filename(self, input_channels):
        """Generate a filename for saving/loading the model based on input channels"""
        ch_str = '_'.join(input_channels)
        return os.path.join(os.path.dirname(__file__), '..', '..', 'models', f"channel_reconstructor_{ch_str}.pt")

    def get_scaler_filename(self, input_channels):
        """Generate a filename for saving/loading the scaler based on input channels"""
        ch_str = '_'.join(input_channels)
        return os.path.join(os.path.dirname(__file__), '..', '..', 'models', f"scaler_{ch_str}.joblib")

    def __init__(self, subjects_dir, ch_names):
        self.subjects_dir = subjects_dir
        self.ch_names = ch_names
        self.fwd = None
        self.src = None
        self.bem = None
        self.scaler = StandardScaler()
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

    def load_all_trial_data(self):
        """Load all trial data from all three baseline conditions for training"""
        print(f"Loading {total_trials} trials from {len(data_conditions)} conditions for training...")

        all_data = []

        for condition_path in data_conditions:
            condition_name = condition_path.split('\\')[-1]
            print(f"Loading condition: {condition_name}")

            for trial in range(1, n_trials_per_condition + 1):
                trial_path = f"{condition_path}\\trial_{trial:02d}\\eeg_data.csv"
                df = pd.read_csv(trial_path)
                data = df.filter(like='eeg').iloc[:, :19].values.T  # Shape: (19, n_times)

                # Transpose to (n_times, n_channels) for easier processing
                all_data.append(data.T)

        # Concatenate all trials from all conditions
        full_data = np.concatenate(all_data, axis=0)  # Shape: (total_samples, 19)
        print(f"Loaded data shape: {full_data.shape} from {len(all_data)} trials")

        return full_data

    def train_channel_reconstructor(self, input_channels):
        """Train the channel reconstruction model using PyTorch"""
        import joblib
        model_filename = self.get_model_filename(input_channels)
        scaler_filename = self.get_scaler_filename(input_channels)
        try:
            model = torch.load(model_filename)
            scaler = joblib.load(scaler_filename)
            print(f"Loaded model from {model_filename} and scaler from {scaler_filename}")
            self.scaler = scaler
            return model, None
        except Exception:
            print(f"Training channel reconstructor: {len(input_channels)} → {len(self.ch_names)} channels")
            full_data = self.load_all_trial_data()
            input_indices = [self.ch_names.index(ch) for ch in input_channels]
            X = full_data[:, input_indices]
            y = full_data[:, :]  # All channels
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)  # Fit only on input channels
            joblib.dump(scaler, scaler_filename)
            # Torch tensors
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
            model = FastChannelReconstructor(len(input_channels), len(self.ch_names))
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
            torch.save(model, model_filename)
            print(f"Saved model to {model_filename} and scaler to {scaler_filename}")
            self.scaler = scaler
            return model, {'loss': loss.item()}

    def reconstruct_channels(self, model, input_channels, test_data):
        """Use trained ML model to reconstruct full channels from subset"""
        import joblib
        scaler_filename = self.get_scaler_filename(input_channels)
        scaler = joblib.load(scaler_filename)
        input_indices = [self.ch_names.index(ch) for ch in input_channels]
        X_test = test_data[:, input_indices]
        X_test_scaled = scaler.transform(X_test)
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            reconstructed = model(X_tensor).numpy()
        return reconstructed

    def reconstruct_channels_biophysical(self, input_channels, test_data):
        """
        Reconstruct missing channels using biophysical forward model:
        1. Compute source reconstruction from available channels
        2. Use forward model to predict what missing channels should measure
        """
        # Get available and missing channel indices
        available_indices = [self.ch_names.index(ch) for ch in input_channels]
        missing_indices = [i for i in range(len(self.ch_names)) if i not in available_indices]

        # Create data with only available channels
        available_data = test_data[:, available_indices]

        # Create info for available channels only
        available_ch_names = [self.ch_names[i] for i in available_indices]
        available_info = mne.create_info(available_ch_names, 100, 'eeg')
        montage = mne.channels.make_standard_montage('standard_1005')
        available_info.set_montage(montage, on_missing='ignore')

        # Compute source reconstruction from available channels
        stc_from_sparse = self.compute_source_reconstruction(available_data, available_info)

        # Use forward model to predict all channels from the sources
        # This gives us what ALL channels should measure for this source activity
        predicted_all_channels = mne.apply_forward(self.fwd, stc_from_sparse, self.fwd['info'])

        # Extract the predicted data for missing channels
        reconstructed_full = np.zeros_like(test_data)

        # Keep original available channels
        reconstructed_full[:, available_indices] = available_data

        # Fill in missing channels with forward model predictions
        reconstructed_full[:, missing_indices] = predicted_all_channels.data[missing_indices, :].T

        return reconstructed_full

    def compute_source_reconstruction(self, data, info=None):
        """Compute source reconstruction for given data"""
        if info is None:
            info = mne.create_info(self.ch_names, 100, 'eeg')
            montage = mne.channels.make_standard_montage('standard_1005')
            info.set_montage(montage, on_missing='ignore')

        raw = mne.io.RawArray(data.T, info)
        raw.set_eeg_reference(ref_channels='average', projection=True)

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

    def _compute_time_series_metrics(self, original, reconstructed):
        """Compute time series metrics for each channel: Pearson correlation, MSE, R2"""
        n_channels = original.shape[0]
        metrics = {}
        for ch in range(n_channels):
            orig_ts = original[ch, :]
            recon_ts = reconstructed[ch, :]
            corr = np.corrcoef(orig_ts, recon_ts)[0, 1] if np.std(orig_ts) > 0 and np.std(recon_ts) > 0 else np.nan
            mse = np.mean((orig_ts - recon_ts) ** 2)
            r2 = 1 - np.sum((orig_ts - recon_ts) ** 2) / np.sum((orig_ts - np.mean(orig_ts)) ** 2)
            metrics[ch] = {'corr': corr, 'mse': mse, 'r2': r2}
        return metrics

    def compare_reconstructions(self, input_channels_list):
        """Compare source reconstructions between original and reconstructed channels using both methods across all conditions"""
        print("\nComparing source reconstructions using both biophysical and ML approaches...")

        # Test on one trial from each condition
        condition_names = ['Silence', 'Audio/Visual', 'Music']
        test_trial = 8  # Use trial 8 from each condition

        results = {}

        for cond_idx, (condition_path, condition_name) in enumerate(zip(data_conditions, condition_names)):
            print(f"\n=== Testing on {condition_name} condition ===")

            # Load test data for this condition
            trial_path = f"{condition_path}\\trial_{test_trial:02d}\\eeg_data.csv"
            df = pd.read_csv(trial_path)
            original_data = df.filter(like='eeg').iloc[:, :19].values.T  # Shape: (19, n_times)

            # Baseline: Original full channels
            print(f"Computing baseline (original 19 channels) for {condition_name}...")
            stc_original = self.compute_source_reconstruction(original_data.T)
            baseline_metrics = self._compute_source_metrics(stc_original)

            condition_results = {
                'original': baseline_metrics,
                'condition_name': condition_name
            }

            # For each input channel configuration
            for input_channels in input_channels_list:
                print(f"\nTesting reconstruction from {len(input_channels)} channels: {input_channels}")

                # Method 1: Biophysical reconstruction
                print("  → Biophysical reconstruction...")
                reconstructed_data_bio = self.reconstruct_channels_biophysical(input_channels, original_data.T)
                stc_bio = self.compute_source_reconstruction(reconstructed_data_bio)
                bio_metrics = self._compute_source_metrics(stc_bio)
                bio_comparison = self._compare_source_reconstructions(stc_original, stc_bio)

                # Method 2: Machine Learning reconstruction
                print("  → ML reconstruction...")
                model, train_metrics = self.train_channel_reconstructor(input_channels)
                reconstructed_data_ml = self.reconstruct_channels(model, input_channels, original_data.T)
                stc_ml = self.compute_source_reconstruction(reconstructed_data_ml)
                ml_metrics = self._compute_source_metrics(stc_ml)
                ml_comparison = self._compare_source_reconstructions(stc_original, stc_ml)

                # Time series metrics for each channel (channel space)
                # Ensure shape is (channels, samples)
                orig_ch_data = original_data.T if original_data.shape[1] == len(self.ch_names) else original_data
                recon_bio_ch_data = reconstructed_data_bio.T if reconstructed_data_bio.shape[1] == len(self.ch_names) else reconstructed_data_bio
                recon_ml_ch_data = reconstructed_data_ml.T if reconstructed_data_ml.shape[1] == len(self.ch_names) else reconstructed_data_ml
                bio_ts_metrics = self._compute_time_series_metrics(orig_ch_data, recon_bio_ch_data)
                ml_ts_metrics = self._compute_time_series_metrics(orig_ch_data, recon_ml_ch_data)
                condition_results[f'reconstructed_{len(input_channels)}ch'] = {
                    'input_channels': input_channels,
                    'biophysical': {
                        'source_metrics': bio_metrics,
                        'comparison': bio_comparison,
                        'reconstructed_data': reconstructed_data_bio,
                        'time_series_metrics': bio_ts_metrics
                    },
                    'machine_learning': {
                        'train_metrics': train_metrics,
                        'source_metrics': ml_metrics,
                        'comparison': ml_comparison,
                        'reconstructed_data': reconstructed_data_ml,
                        'model': model,
                        'time_series_metrics': ml_ts_metrics
                    }
                }

            results[condition_name] = condition_results

        return results, stc_original

    def _compute_source_metrics(self, stc):
        """Compute metrics for source reconstruction"""
        data = stc.data

        return {
            'peak_activation': np.max(np.abs(data)),
            'mean_activation': np.mean(np.abs(data)),
            'active_sources': np.sum(np.abs(data) > np.percentile(np.abs(data), 95)),
            'total_sources': data.shape[0],
            'spatial_variance': np.var(data, axis=0).mean()
        }

    def _compare_source_reconstructions(self, stc_original, stc_reconstructed):
        """Compare two source reconstructions"""
        orig_data = stc_original.data
        recon_data = stc_reconstructed.data

        # Spatial correlation at each time point
        correlations = []
        for t in range(min(orig_data.shape[1], recon_data.shape[1])):
            if np.std(orig_data[:, t]) > 0 and np.std(recon_data[:, t]) > 0:
                corr = np.corrcoef(orig_data[:, t], recon_data[:, t])[0, 1]
                correlations.append(corr)

        spatial_correlation = np.mean(correlations) if correlations else 0

        # MSE between reconstructions
        min_time = min(orig_data.shape[1], recon_data.shape[1])
        mse = mean_squared_error(orig_data[:, :min_time].flatten(),
                                recon_data[:, :min_time].flatten())

        # Peak location similarity (find top N peaks in both)
        n_peaks = 100
        orig_peaks = np.argsort(np.abs(orig_data).max(axis=1))[-n_peaks:]
        recon_peaks = np.argsort(np.abs(recon_data).max(axis=1))[-n_peaks:]

        peak_overlap = len(set(orig_peaks) & set(recon_peaks)) / n_peaks

        return {
            'spatial_correlation': spatial_correlation,
            'mse': mse,
            'peak_overlap': peak_overlap
        }

def plot_comparison_results(results):
    """Plot the comparison results for both biophysical and ML reconstruction across conditions"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    condition_names = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green for conditions

    # For each condition
    for cond_idx, condition_name in enumerate(condition_names):
        condition_results = results[condition_name]

        # Source reconstruction comparison - both methods for this condition
        configs = []
        bio_corr = []
        ml_corr = []
        bio_peak = []
        ml_peak = []

        for key, result in condition_results.items():
            if key.startswith('reconstructed'):
                n_ch = len(result['input_channels'])
                configs.append(f'{n_ch}ch')

                bio_corr.append(result['biophysical']['comparison']['spatial_correlation'])
                ml_corr.append(result['machine_learning']['comparison']['spatial_correlation'])

                bio_peak.append(result['biophysical']['comparison']['peak_overlap'])
                ml_peak.append(result['machine_learning']['comparison']['peak_overlap'])

        x = np.arange(len(configs))
        width = 0.35

        # Spatial correlation
        axes[0, cond_idx].bar(x - width/2, bio_corr, width, label='Biophysical', alpha=0.8, color='skyblue')
        axes[0, cond_idx].bar(x + width/2, ml_corr, width, label='ML', alpha=0.8, color='lightcoral')
        axes[0, cond_idx].set_title(f'{condition_name}\nSpatial Correlation')
        axes[0, cond_idx].set_ylabel('Correlation with Original')
        axes[0, cond_idx].set_xticks(x)
        axes[0, cond_idx].set_xticklabels(configs)
        if cond_idx == 0:
            axes[0, cond_idx].legend()
        axes[0, cond_idx].grid(True, alpha=0.3)

        # Peak overlap
        axes[1, cond_idx].bar(x - width/2, bio_peak, width, label='Biophysical', alpha=0.8, color='skyblue')
        axes[1, cond_idx].bar(x + width/2, ml_peak, width, label='ML', alpha=0.8, color='lightcoral')
        axes[1, cond_idx].set_title(f'{condition_name}\nPeak Activation Overlap')
        axes[1, cond_idx].set_ylabel('Overlap Fraction')
        axes[1, cond_idx].set_xticks(x)
        axes[1, cond_idx].set_xticklabels(configs)
        axes[1, cond_idx].grid(True, alpha=0.3)

        # Save per-condition plot: Biophysical vs ML (spatial correlation)
        fig_both, ax_both = plt.subplots(figsize=(7, 5))
        ax_both.bar(x - width/2, bio_corr, width, label='Biophysical', alpha=0.8, color='skyblue')
        ax_both.bar(x + width/2, ml_corr, width, label='ML', alpha=0.8, color='lightcoral')
        ax_both.set_title(f'{condition_name}\nSpatial Correlation: Biophysical vs ML')
        ax_both.set_ylabel('Correlation with Original')
        ax_both.set_xticks(x)
        ax_both.set_xticklabels(configs)
        ax_both.legend()
        ax_both.grid(True, alpha=0.3)
        plt.tight_layout()
        both_fig_name = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', f'comparison_spatialcorr_{condition_name.replace(" ", "_").replace("/", "_").lower()}.png')
        plt.savefig(both_fig_name, dpi=300, bbox_inches='tight')
        plt.close(fig_both)
        # Save per-condition plot: Biophysical vs ML (peak overlap)
        fig_both_peak, ax_both_peak = plt.subplots(figsize=(7, 5))
        ax_both_peak.bar(x - width/2, bio_peak, width, label='Biophysical', alpha=0.8, color='skyblue')
        ax_both_peak.bar(x + width/2, ml_peak, width, label='ML', alpha=0.8, color='lightcoral')
        ax_both_peak.set_title(f'{condition_name}\nPeak Activation Overlap: Biophysical vs ML')
        ax_both_peak.set_ylabel('Overlap Fraction')
        ax_both_peak.set_xticks(x)
        ax_both_peak.set_xticklabels(configs)
        ax_both_peak.legend()
        ax_both_peak.grid(True, alpha=0.3)
        plt.tight_layout()
        both_peak_fig_name = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', f'comparison_peakoverlap_{condition_name.replace(" ", "_").replace("/", "_").lower()}.png')
        plt.savefig(both_peak_fig_name, dpi=300, bbox_inches='tight')
        plt.close(fig_both_peak)

        # Time series correlation per channel
        for key, result in condition_results.items():
            if key.startswith('reconstructed'):
                n_ch = len(result['input_channels'])
                bio_ts_metrics = result['biophysical']['time_series_metrics']
                ml_ts_metrics = result['machine_learning']['time_series_metrics']
                # Plot time series correlation for all channels
                fig_ts, ax_ts = plt.subplots(figsize=(8, 5))
                bio_corrs = [bio_ts_metrics[ch]['corr'] for ch in range(len(ch_names))]
                ml_corrs = [ml_ts_metrics[ch]['corr'] for ch in range(len(ch_names))]
                ax_ts.plot(ch_names, bio_corrs, marker='o', label='Biophysical', color='skyblue')
                ax_ts.plot(ch_names, ml_corrs, marker='o', label='ML', color='lightcoral')
                ax_ts.set_title(f'{condition_name} - {n_ch}ch\nTime Series Correlation per Channel')
                ax_ts.set_ylabel('Pearson Correlation')
                ax_ts.set_xticks(range(len(ch_names)))
                ax_ts.set_xticklabels(ch_names, rotation=45)
                ax_ts.legend()
                ax_ts.grid(True, alpha=0.3)
                plt.tight_layout()
    
                ts_fig_name = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', f'timeseries_corr_{condition_name.replace(" ", "_").replace("/", "_").lower()}_{n_ch}ch.png')
                plt.savefig(ts_fig_name, dpi=300, bbox_inches='tight')
                plt.close(fig_ts)
                # Plot R² score for all channels
                fig_r2, ax_r2 = plt.subplots(figsize=(8, 5))
                bio_r2 = [bio_ts_metrics[ch]['r2'] for ch in range(len(ch_names))]
                ml_r2 = [ml_ts_metrics[ch]['r2'] for ch in range(len(ch_names))]
                ax_r2.plot(ch_names, bio_r2, marker='o', label='Biophysical', color='skyblue')
                ax_r2.plot(ch_names, ml_r2, marker='o', label='ML', color='lightcoral')
                ax_r2.set_title(f'{condition_name} - {n_ch}ch\nTime Series R² per Channel')
                ax_r2.set_ylabel('R² Score')
                ax_r2.set_xticks(range(len(ch_names)))
                ax_r2.set_xticklabels(ch_names, rotation=45)
                ax_r2.legend()
                ax_r2.grid(True, alpha=0.3)
                plt.tight_layout()
    
                r2_fig_name = os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', f'timeseries_r2_{condition_name.replace(" ", "_").replace("/", "_").lower()}_{n_ch}ch.png')
                plt.savefig(r2_fig_name, dpi=300, bbox_inches='tight')
                plt.close(fig_r2)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', '..', 'outputs', 'channel_reconstruction_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n" + "="*80)
    print("CROSS-CONDITION RELIABILITY ANALYSIS SUMMARY")
    print("="*80)

    for condition_name in condition_names:
        print(f"\n{condition_name.upper()} CONDITION:")
        condition_results = results[condition_name]

        for key, result in condition_results.items():
            if key.startswith('reconstructed'):
                n_ch = len(result['input_channels'])
                print(f"\n  {n_ch} Channel Reconstruction:")

                bio_corr = result['biophysical']['comparison']['spatial_correlation']
                ml_corr = result['machine_learning']['comparison']['spatial_correlation']
                bio_peak = result['biophysical']['comparison']['peak_overlap']
                ml_peak = result['machine_learning']['comparison']['peak_overlap']

                print(".4f")
                print(".4f")
                print(".4f")
                print(".4f")

                # Determine which method performed better
                if bio_corr > ml_corr:
                    print("    → Biophysical method superior for spatial accuracy")
                else:
                    print("    → ML method superior for spatial accuracy")

    # Cross-condition consistency analysis
    print(f"\n{'='*80}")
    print("CROSS-CONDITION CONSISTENCY ANALYSIS")
    print(f"{'='*80}")

    # Compare performance across conditions for each method and channel count
    for n_ch in [5, 10, 15]:
        print(f"\n{n_ch} Channel Reconstruction Across Conditions:")

        bio_corrs = []
        ml_corrs = []

        for condition_name in condition_names:
            condition_results = results[condition_name]
            result_key = f'reconstructed_{n_ch}ch'

            if result_key in condition_results:
                bio_corr = condition_results[result_key]['biophysical']['comparison']['spatial_correlation']
                ml_corr = condition_results[result_key]['machine_learning']['comparison']['spatial_correlation']

                bio_corrs.append(bio_corr)
                ml_corrs.append(ml_corr)

        if bio_corrs and ml_corrs:
            bio_std = np.std(bio_corrs)
            ml_std = np.std(ml_corrs)
            bio_mean = np.mean(bio_corrs)
            ml_mean = np.mean(ml_corrs)

            print(".4f")
            print(".4f")
            print(".4f")
            print(".4f")

            if bio_std < ml_std:
                print(f"    → Biophysical method more consistent across conditions (lower std: {bio_std:.4f} vs {ml_std:.4f})")
            else:
                print(f"    → ML method more consistent across conditions (lower std: {ml_std:.4f} vs {bio_std:.4f})")

# Main analysis
if __name__ == "__main__":
    print("Starting Channel Reconstruction Analysis...")

    # Suppress MNE verbose output
    mne.set_log_level('WARNING')

    analyzer = ChannelReconstructionAnalyzer(subjects_dir, ch_names)

    # Import top channels from importance analysis
    from channel_importance_analysis import ChannelImportanceAnalyzer

    # Recompute combined variance importance to get top channels
    subjects_dir = r"C:\Users\unnat\mne_data\MNE-fsaverage-data"
    ch_names = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','O2']
    base_paths = [
        r"C:\Users\unnat\Desktop\EEGFeatureExtraction\Subject_1\Baseline (in_silence)",
        r"C:\Users\unnat\Desktop\EEGFeatureExtraction\Subject_1\Baseline (with_audio_and_visual_stimulus)",
        r"C:\Users\unnat\Desktop\EEGFeatureExtraction\Subject_1\Baseline (with_music)"
    ]
    n_trials = 20
    importance_analyzer = ChannelImportanceAnalyzer(subjects_dir, ch_names)
    variance_importance = importance_analyzer._channel_variance_importance(base_paths, n_trials)
    top_15 = list(variance_importance.keys())[:15]
    top_10 = list(variance_importance.keys())[:10]
    top_5 = list(variance_importance.keys())[:5]

    # Test different input channel configurations using top channels
    input_configs = [
        top_5,
        top_10,
        top_15
    ]

    # Compare reconstructions across conditions
    results, stc_original = analyzer.compare_reconstructions(input_configs)

    # Plot results (this now includes the detailed summary)
    plot_comparison_results(results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("✓ Tested biophysical vs ML channel reconstruction across 3 conditions")
    print("✓ Used meta-analysis validated tissue conductivity parameters")
    print("✓ Generated comprehensive comparison plots and statistics")
    print("✓ Biophysical: Physics-based head model | ML: Data-driven correlations")