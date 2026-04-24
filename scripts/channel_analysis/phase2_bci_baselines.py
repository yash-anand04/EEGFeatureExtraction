import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

try:
    from mne.channels import make_standard_montage
    from mne.channels.interpolation import _make_interpolation_matrix
except Exception:
    make_standard_montage = None
    _make_interpolation_matrix = None


CHANNELS_22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]

# Approximate 2D scalp coordinates (top view) aligned to 10-10 relative placement.
# These coordinates are used by distance-weighted interpolation and visualization.
CHANNEL_POS_2D = {
    "Fz": (0.00, 0.62),
    "FC3": (-0.33, 0.34),
    "FC1": (-0.14, 0.34),
    "FCz": (0.00, 0.32),
    "FC2": (0.14, 0.34),
    "FC4": (0.33, 0.34),
    "C5": (-0.50, 0.00),
    "C3": (-0.33, 0.00),
    "C1": (-0.14, 0.00),
    "Cz": (0.00, 0.00),
    "C2": (0.14, 0.00),
    "C4": (0.33, 0.00),
    "C6": (0.50, 0.00),
    "CP3": (-0.33, -0.30),
    "CP1": (-0.14, -0.30),
    "CPz": (0.00, -0.30),
    "CP2": (0.14, -0.30),
    "CP4": (0.33, -0.30),
    "P1": (-0.14, -0.52),
    "Pz": (0.00, -0.54),
    "P2": (0.14, -0.52),
    "POz": (0.00, -0.82),
}

# Adapted channel subsets for this 22-channel montage.
# (Roadmap had Oz/T7, which are not in BCI IV-2a 22-channel EEG list.)
CHANNEL_SET_5 = ["Fz", "C3", "C4", "Pz", "POz"]
CHANNEL_SET_10 = ["Fz", "FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "Pz", "POz"]
CHANNEL_SET_15 = ["Fz", "FC3", "FC1", "FC2", "FC4", "C3", "C1", "Cz", "C2", "C4", "CP3", "CP1", "CP2", "CP4", "Pz"]

_SPLINE_MATRIX_CACHE = {}


def _assert_channels(chs):
    missing = [c for c in chs if c not in CHANNELS_22]
    if missing:
        raise ValueError(f"Unknown channels in set: {missing}")


def load_subject_data(subject_dir: Path):
    npz = np.load(subject_dir / "trials.npz", allow_pickle=True)
    x = npz["X"].astype(np.float32)  # (n_trials, n_channels, n_times)
    y = npz["y"].astype(np.int32)
    return x, y


def to_sample_matrix(trials_ch_time: np.ndarray):
    # (n_trials, n_channels, n_times) -> (n_trials*n_times, n_channels)
    n_trials, n_channels, n_times = trials_ch_time.shape
    return np.transpose(trials_ch_time, (0, 2, 1)).reshape(n_trials * n_times, n_channels)


def from_sample_matrix(samples_ch: np.ndarray, n_trials: int, n_times: int):
    # (n_trials*n_times, n_channels) -> (n_trials, n_channels, n_times)
    n_channels = samples_ch.shape[1]
    out = samples_ch.reshape(n_trials, n_times, n_channels)
    return np.transpose(out, (0, 2, 1))


def distance_weighted_interpolation(X_input: np.ndarray, input_channels, target_channels):
    """Interpolate missing channels using inverse-distance weighting in 2D scalp space.

    X_input shape: (n_samples, n_input_channels)
    returns: (n_samples, n_target_channels)
    """
    eps = 1e-8
    input_pos = np.array([CHANNEL_POS_2D[c] for c in input_channels], dtype=np.float32)

    preds = []
    for t_ch in target_channels:
        t_pos = np.array(CHANNEL_POS_2D[t_ch], dtype=np.float32)
        d = np.linalg.norm(input_pos - t_pos[None, :], axis=1)
        w = 1.0 / (d + eps)
        w = w / np.sum(w)
        preds.append((X_input * w[None, :]).sum(axis=1))

    return np.stack(preds, axis=1)


def _get_spherical_positions(channels):
    if make_standard_montage is None:
        raise ImportError(
            "mne is required for spherical spline interpolation. Install with: pip install mne"
        )
    montage = make_standard_montage("standard_1005")
    ch_pos = montage.get_positions()["ch_pos"]
    pos = []
    for ch in channels:
        if ch not in ch_pos:
            raise ValueError(f"Channel {ch} not available in standard_1005 montage")
        p = np.asarray(ch_pos[ch], dtype=np.float64)
        p = p / max(np.linalg.norm(p), 1e-12)
        pos.append(p)
    return np.stack(pos, axis=0)


def spherical_spline_interpolation(X_input: np.ndarray, input_channels, target_channels):
    """Interpolate missing channels using Perrin spherical spline interpolation.

    Uses MNE's interpolation matrix implementation.
    """
    if _make_interpolation_matrix is None:
        raise ImportError(
            "mne is required for spherical spline interpolation. Install with: pip install mne"
        )

    key = (tuple(input_channels), tuple(target_channels))
    if key not in _SPLINE_MATRIX_CACHE:
        pos_from = _get_spherical_positions(input_channels)
        pos_to = _get_spherical_positions(target_channels)
        _SPLINE_MATRIX_CACHE[key] = _make_interpolation_matrix(pos_from, pos_to)

    # matrix shape: (n_targets, n_inputs)
    interp_mat = _SPLINE_MATRIX_CACHE[key]
    return X_input @ interp_mat.T


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    diff = y_true - y_pred
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)

    yt_c = yt - yt.mean()
    yp_c = yp - yp.mean()
    den = np.sqrt(np.sum(yt_c ** 2) * np.sum(yp_c ** 2))
    corr = float(np.sum(yt_c * yp_c) / den) if den > 0 else np.nan

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "pearson_r": corr,
        "r2": r2,
    }


def _safe_corr(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    a_c = a - a.mean()
    b_c = b - b.mean()
    den = np.sqrt(np.sum(a_c ** 2) * np.sum(b_c ** 2))
    if den <= 0:
        return np.nan
    return float(np.sum(a_c * b_c) / den)


def _band_power(psd: np.ndarray, freqs: np.ndarray, f_low: float, f_high: float):
    mask = (freqs >= f_low) & (freqs < f_high)
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=np.float64)
    return np.trapezoid(psd[:, mask], freqs[mask], axis=1)


def compute_advanced_metrics(y_true: np.ndarray, y_pred: np.ndarray, sfreq: int = 250):
    """Compute spatial and frequency metrics for roadmap Phase 4.

    y_true, y_pred shapes: (n_samples, n_channels)
    """
    # Spatial topography correlation from channel RMS maps.
    true_rms = np.sqrt(np.mean(y_true ** 2, axis=0))
    pred_rms = np.sqrt(np.mean(y_pred ** 2, axis=0))
    spatial_topo_corr = _safe_corr(true_rms, pred_rms)

    # Peak node overlap: channel with max abs activation per sample.
    true_peak_idx = np.argmax(np.abs(y_true), axis=1)
    pred_peak_idx = np.argmax(np.abs(y_pred), axis=1)
    peak_node_overlap = float(np.mean(true_peak_idx == pred_peak_idx))

    # PSD-based band errors averaged across channels.
    # Treat each channel as one long signal over concatenated samples.
    freqs, psd_true = welch(y_true.T, fs=sfreq, nperseg=min(512, y_true.shape[0]), axis=1)
    _, psd_pred = welch(y_pred.T, fs=sfreq, nperseg=min(512, y_pred.shape[0]), axis=1)

    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }

    out = {
        "spatial_topo_corr": spatial_topo_corr,
        "peak_node_overlap": peak_node_overlap,
    }

    for band_name, (f_l, f_h) in bands.items():
        bp_true = _band_power(psd_true, freqs, f_l, f_h)
        bp_pred = _band_power(psd_pred, freqs, f_l, f_h)
        out[f"{band_name}_band_rmse"] = float(np.sqrt(np.mean((bp_true - bp_pred) ** 2)))

    return out


def run_split(train_subjects, test_subject, processed_dir: Path, input_channels, max_train_samples: int, rng: np.random.Generator):
    _assert_channels(input_channels)

    input_idx = [CHANNELS_22.index(c) for c in input_channels]
    missing_channels = [c for c in CHANNELS_22 if c not in input_channels]
    missing_idx = [CHANNELS_22.index(c) for c in missing_channels]

    # Build train sample matrix from train subjects.
    train_blocks = []
    for subj in train_subjects:
        x_trials, _ = load_subject_data(processed_dir / subj)
        train_blocks.append(to_sample_matrix(x_trials))
    train_samples = np.concatenate(train_blocks, axis=0)

    if max_train_samples > 0 and train_samples.shape[0] > max_train_samples:
        sel = rng.choice(train_samples.shape[0], size=max_train_samples, replace=False)
        train_samples = train_samples[sel]

    X_train = train_samples[:, input_idx]
    Y_train_missing = train_samples[:, missing_idx]

    # Test data
    x_test_trials, _ = load_subject_data(processed_dir / test_subject)
    n_trials, _, n_times = x_test_trials.shape
    test_samples = to_sample_matrix(x_test_trials)

    X_test_input = test_samples[:, input_idx]
    Y_test_missing = test_samples[:, missing_idx]

    # Baseline 1: distance-weighted interpolation (non-trainable).
    pred_interp_missing = distance_weighted_interpolation(X_test_input, input_channels, missing_channels)

    # Baseline 1b: spherical spline interpolation (non-trainable).
    pred_spline_missing = spherical_spline_interpolation(X_test_input, input_channels, missing_channels)

    # Baseline 2: pure ML linear model.
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, Y_train_missing)
    pred_ridge_missing = ridge.predict(X_test_input)

    # Baseline 3: pure AI MLP.
    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=80,
        random_state=42,
        early_stopping=True,
        n_iter_no_change=8,
        validation_fraction=0.1,
    )
    mlp.fit(X_train, Y_train_missing)
    pred_mlp_missing = mlp.predict(X_test_input)

    results = []
    for method, pred in [
        ("distance_weighted_interpolation", pred_interp_missing),
        ("spherical_spline_interpolation", pred_spline_missing),
        ("ridge_regression", pred_ridge_missing),
        ("mlp_regressor", pred_mlp_missing),
    ]:
        metrics = compute_metrics(Y_test_missing, pred)
        metrics.update(compute_advanced_metrics(Y_test_missing, pred, sfreq=250))
        results.append(
            {
                "test_subject": test_subject,
                "n_input_channels": len(input_channels),
                "input_channels": ",".join(input_channels),
                "n_reconstructed_channels": len(missing_channels),
                "method": method,
                **metrics,
                "n_test_trials": int(n_trials),
                "n_test_samples": int(n_trials * n_times),
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2 baseline benchmark on preprocessed BCI IV-2a dataset")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a",
        help="Directory containing subject NPZ files and loso_splits.json",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=200000,
        help="Cap train sample rows for speed; set <=0 to disable cap",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--max-splits",
        type=int,
        default=0,
        help="Limit number of LOSO splits to run (0 means all)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="distance_weighted_interpolation,spherical_spline_interpolation,ridge_regression,mlp_regressor",
        help="Comma-separated methods to run",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs") / "phase2_bci_baselines_loso.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=Path("outputs") / "phase2_bci_baselines_summary.json",
        help="Output summary JSON path",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir.resolve()
    loso_path = processed_dir / "loso_splits.json"
    if not loso_path.exists():
        raise FileNotFoundError(f"Missing LOSO splits file: {loso_path}")

    loso_splits = json.loads(loso_path.read_text(encoding="utf-8"))
    if args.max_splits > 0:
        loso_splits = loso_splits[: args.max_splits]

    selected_methods = {m.strip() for m in args.methods.split(",") if m.strip()}

    channel_sets = [CHANNEL_SET_5, CHANNEL_SET_10, CHANNEL_SET_15]
    rng = np.random.default_rng(args.seed)

    all_results = []
    for split in loso_splits:
        test_subject = split["test_subject"]
        train_subjects = split["train_subjects"]
        for ch_set in channel_sets:
            split_results = run_split(
                train_subjects=train_subjects,
                test_subject=test_subject,
                processed_dir=processed_dir,
                input_channels=ch_set,
                max_train_samples=args.max_train_samples,
                rng=rng,
            )
            for row in split_results:
                if row["method"] in selected_methods:
                    all_results.append(row)

    if not all_results:
        raise RuntimeError("No results generated. Check --methods selection.")

    out_df = pd.DataFrame(all_results)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_csv.resolve()
    out_df.to_csv(out_csv, index=False)

    metric_cols = [
        c
        for c in out_df.columns
        if c not in {"test_subject", "n_input_channels", "input_channels", "n_reconstructed_channels", "method", "n_test_trials", "n_test_samples"}
    ]
    summary = out_df.groupby(["n_input_channels", "method"], as_index=False)[metric_cols].mean().sort_values(
        ["n_input_channels", "method"]
    )

    summary_payload = {
        "rows": int(len(out_df)),
        "splits": int(len(loso_splits)),
        "channel_sets": {
            "5": CHANNEL_SET_5,
            "10": CHANNEL_SET_10,
            "15": CHANNEL_SET_15,
        },
        "note": "Input channel sets are adapted to available BCI IV-2a channels.",
        "mean_metrics": summary.to_dict(orient="records"),
    }

    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    out_json = args.output_summary_json.resolve()
    out_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Saved detailed results: {out_csv}")
    print(f"Saved summary: {out_json}")


if __name__ == "__main__":
    main()
