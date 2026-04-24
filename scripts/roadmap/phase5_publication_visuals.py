import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.channel_analysis.phase2_bci_baselines import (
    CHANNELS_22,
    CHANNEL_POS_2D,
    CHANNEL_SET_10,
    distance_weighted_interpolation,
    load_subject_data,
    to_sample_matrix,
)


def _train_models(train_samples: np.ndarray, input_channels: list[str], max_train_samples: int, seed: int):
    rng = np.random.default_rng(seed)

    input_idx = [CHANNELS_22.index(c) for c in input_channels]
    missing_channels = [c for c in CHANNELS_22 if c not in input_channels]
    missing_idx = [CHANNELS_22.index(c) for c in missing_channels]

    if max_train_samples > 0 and train_samples.shape[0] > max_train_samples:
        sel = rng.choice(train_samples.shape[0], size=max_train_samples, replace=False)
        train_samples = train_samples[sel]

    x_train = train_samples[:, input_idx]
    y_train_missing = train_samples[:, missing_idx]

    y_interp_train = distance_weighted_interpolation(x_train, input_channels, missing_channels)

    ridge = Ridge(alpha=1.0)
    ridge.fit(x_train, y_train_missing)

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=80,
        random_state=seed,
        early_stopping=True,
        n_iter_no_change=8,
        validation_fraction=0.1,
    )
    mlp.fit(x_train, y_train_missing)

    y_residual = y_train_missing - y_interp_train
    x_h_train = np.concatenate([x_train, y_interp_train], axis=1)
    hybrid_ridge = Ridge(alpha=1.0)
    hybrid_ridge.fit(x_h_train, y_residual)

    return {
        "ridge": ridge,
        "mlp": mlp,
        "hybrid_ridge": hybrid_ridge,
        "missing_channels": missing_channels,
        "input_idx": input_idx,
        "missing_idx": missing_idx,
    }


def _predict_full_methods(test_samples: np.ndarray, input_channels: list[str], model_bundle: dict):
    missing_channels = model_bundle["missing_channels"]
    input_idx = model_bundle["input_idx"]
    missing_idx = model_bundle["missing_idx"]

    x_test = test_samples[:, input_idx]
    y_true_missing = test_samples[:, missing_idx]

    pred_interp = distance_weighted_interpolation(x_test, input_channels, missing_channels)
    pred_ridge = model_bundle["ridge"].predict(x_test)
    pred_mlp = model_bundle["mlp"].predict(x_test)

    x_h_test = np.concatenate([x_test, pred_interp], axis=1)
    pred_hybrid = pred_interp + model_bundle["hybrid_ridge"].predict(x_h_test)

    preds_missing = {
        "distance_weighted_interpolation": pred_interp,
        "ridge_regression": pred_ridge,
        "mlp_regressor": pred_mlp,
        "hybrid_residual_ridge": pred_hybrid,
    }

    full_preds = {}
    for method, y_pred_missing in preds_missing.items():
        arr = np.zeros((test_samples.shape[0], len(CHANNELS_22)), dtype=np.float32)
        arr[:, input_idx] = x_test
        arr[:, missing_idx] = y_pred_missing
        full_preds[method] = arr

    return full_preds, y_true_missing, missing_channels


def _plot_timeseries_overlay(y_true_full: np.ndarray, preds: dict[str, np.ndarray], channel: str, out_path: Path):
    ch_idx = CHANNELS_22.index(channel)
    n_show = min(1250, y_true_full.shape[0])

    plt.figure(figsize=(12, 5))
    plt.plot(y_true_full[:n_show, ch_idx], label="ground_truth", linewidth=2, color="black")
    for method, pred in preds.items():
        plt.plot(pred[:n_show, ch_idx], label=method, alpha=0.9)

    plt.title(f"Time-Series Overlay at {channel} (first {n_show} samples)")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_topomap_errors(y_true_full: np.ndarray, preds: dict[str, np.ndarray], out_path: Path):
    methods = list(preds.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    xs = np.array([CHANNEL_POS_2D[ch][0] for ch in CHANNELS_22])
    ys = np.array([CHANNEL_POS_2D[ch][1] for ch in CHANNELS_22])

    for i, method in enumerate(methods):
        mae_ch = np.mean(np.abs(y_true_full - preds[method]), axis=0)
        ax = axes[i]
        sc = ax.scatter(xs, ys, c=mae_ch, s=220, cmap="plasma", edgecolor="black", linewidth=0.4)
        for j, ch in enumerate(CHANNELS_22):
            ax.text(xs[j], ys[j] + 0.035, ch, fontsize=7, ha="center")
        ax.set_title(f"MAE Topography: {method}")
        ax.set_xlim(-0.65, 0.65)
        ax.set_ylim(-0.95, 0.75)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Topographic Channel Error Maps")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_psd_comparison(y_true_full: np.ndarray, preds: dict[str, np.ndarray], channel: str, out_path: Path, sfreq: int):
    ch_idx = CHANNELS_22.index(channel)

    plt.figure(figsize=(11, 5))

    f_true, psd_true = welch(y_true_full[:, ch_idx], fs=sfreq, nperseg=min(512, y_true_full.shape[0]))
    plt.semilogy(f_true, psd_true, linewidth=2, color="black", label="ground_truth")

    for method, pred in preds.items():
        f_m, psd_m = welch(pred[:, ch_idx], fs=sfreq, nperseg=min(512, pred.shape[0]))
        plt.semilogy(f_m, psd_m, label=method)

    plt.xlim(0.5, 45)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(f"PSD Comparison at {channel}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_error_topology(y_true_full: np.ndarray, preds: dict[str, np.ndarray], out_path: Path):
    methods = list(preds.keys())
    err_mat = []
    for method in methods:
        mae_ch = np.mean(np.abs(y_true_full - preds[method]), axis=0)
        err_mat.append(mae_ch)
    err = np.array(err_mat)

    plt.figure(figsize=(13, 5))
    im = plt.imshow(err, aspect="auto", cmap="magma")
    plt.yticks(np.arange(len(methods)), methods)
    plt.xticks(np.arange(len(CHANNELS_22)), CHANNELS_22, rotation=60, fontsize=8)
    plt.colorbar(im, label="MAE")
    plt.title("Electrode-Level Error Topology (Method x Channel)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate publication-style Phase 5 visuals from a LOSO split")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a",
    )
    parser.add_argument(
        "--split-index",
        type=int,
        default=0,
        help="Index in loso_splits.json to visualize",
    )
    parser.add_argument(
        "--input-channels",
        type=str,
        default=",".join(CHANNEL_SET_10),
        help="Comma-separated input channels",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=60000,
    )
    parser.add_argument(
        "--target-channel",
        type=str,
        default="POz",
    )
    parser.add_argument("--sfreq", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase5_publication_plots",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir.resolve()
    loso_splits = json.loads((processed_dir / "loso_splits.json").read_text(encoding="utf-8"))

    if args.split_index < 0 or args.split_index >= len(loso_splits):
        raise ValueError(f"split-index must be in [0, {len(loso_splits)-1}]")

    split = loso_splits[args.split_index]
    test_subject = split["test_subject"]
    train_subjects = split["train_subjects"]
    input_channels = [c.strip() for c in args.input_channels.split(",") if c.strip()]

    for ch in input_channels:
        if ch not in CHANNELS_22:
            raise ValueError(f"Unknown input channel: {ch}")

    if args.target_channel not in CHANNELS_22:
        raise ValueError(f"Unknown target-channel: {args.target_channel}")

    train_blocks = []
    for subj in train_subjects:
        x_trials, _ = load_subject_data(processed_dir / subj)
        train_blocks.append(to_sample_matrix(x_trials))
    train_samples = np.concatenate(train_blocks, axis=0)

    x_test_trials, _ = load_subject_data(processed_dir / test_subject)
    test_samples = to_sample_matrix(x_test_trials)

    model_bundle = _train_models(
        train_samples=train_samples,
        input_channels=input_channels,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
    )
    preds_full, _, _ = _predict_full_methods(
        test_samples=test_samples,
        input_channels=input_channels,
        model_bundle=model_bundle,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _plot_timeseries_overlay(
        y_true_full=test_samples,
        preds=preds_full,
        channel=args.target_channel,
        out_path=args.out_dir / "timeseries_overlay.png",
    )
    _plot_topomap_errors(
        y_true_full=test_samples,
        preds=preds_full,
        out_path=args.out_dir / "topomap_error_maps.png",
    )
    _plot_psd_comparison(
        y_true_full=test_samples,
        preds=preds_full,
        channel=args.target_channel,
        out_path=args.out_dir / "psd_comparison.png",
        sfreq=args.sfreq,
    )
    _plot_error_topology(
        y_true_full=test_samples,
        preds=preds_full,
        out_path=args.out_dir / "electrode_error_topology.png",
    )

    summary = {
        "split_index": int(args.split_index),
        "test_subject": test_subject,
        "input_channels": input_channels,
        "target_channel": args.target_channel,
        "methods": list(preds_full.keys()),
    }
    (args.out_dir / "publication_visual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved publication-style Phase 5 plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
