import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.channel_analysis.phase2_bci_baselines import (
    CHANNELS_22,
    CHANNEL_SET_5,
    CHANNEL_SET_10,
    CHANNEL_SET_15,
    compute_advanced_metrics,
    compute_metrics,
    distance_weighted_interpolation,
    load_subject_data,
    to_sample_matrix,
)


def _assert_channels(chs):
    missing = [c for c in chs if c not in CHANNELS_22]
    if missing:
        raise ValueError(f"Unknown channels in set: {missing}")


def _fit_residual_model(model_name: str, x_train: np.ndarray, y_train: np.ndarray):
    if model_name == "ridge":
        model = Ridge(alpha=1.0)
    elif model_name == "mlp":
        model = MLPRegressor(
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
    else:
        raise ValueError(f"Unsupported residual model: {model_name}")

    model.fit(x_train, y_train)
    return model


def run_split(train_subjects, test_subject, processed_dir: Path, input_channels, max_train_samples: int, rng, residual_model: str):
    _assert_channels(input_channels)

    input_idx = [CHANNELS_22.index(c) for c in input_channels]
    missing_channels = [c for c in CHANNELS_22 if c not in input_channels]
    missing_idx = [CHANNELS_22.index(c) for c in missing_channels]

    train_blocks = []
    for subj in train_subjects:
        x_trials, _ = load_subject_data(processed_dir / subj)
        train_blocks.append(to_sample_matrix(x_trials))
    train_samples = np.concatenate(train_blocks, axis=0)

    if max_train_samples > 0 and train_samples.shape[0] > max_train_samples:
        sel = rng.choice(train_samples.shape[0], size=max_train_samples, replace=False)
        train_samples = train_samples[sel]

    x_train_input = train_samples[:, input_idx]
    y_train_missing = train_samples[:, missing_idx]

    # Physics-inspired baseline (distance-weighted interpolation)
    y_train_interp = distance_weighted_interpolation(x_train_input, input_channels, missing_channels)

    # Residual target
    y_train_residual = y_train_missing - y_train_interp

    # Hybrid feature: concatenate available channels + interpolation estimate
    x_train_hybrid = np.concatenate([x_train_input, y_train_interp], axis=1)

    model = _fit_residual_model(residual_model, x_train_hybrid, y_train_residual)

    # Test data
    x_test_trials, _ = load_subject_data(processed_dir / test_subject)
    n_trials, _, n_times = x_test_trials.shape
    test_samples = to_sample_matrix(x_test_trials)
    x_test_input = test_samples[:, input_idx]
    y_test_missing = test_samples[:, missing_idx]

    y_test_interp = distance_weighted_interpolation(x_test_input, input_channels, missing_channels)
    x_test_hybrid = np.concatenate([x_test_input, y_test_interp], axis=1)
    y_pred_residual = model.predict(x_test_hybrid)

    y_pred_hybrid = y_test_interp + y_pred_residual

    out_rows = []
    for method_name, pred in [
        ("distance_weighted_interpolation", y_test_interp),
        (f"hybrid_residual_{residual_model}", y_pred_hybrid),
    ]:
        metrics = compute_metrics(y_test_missing, pred)
        metrics.update(compute_advanced_metrics(y_test_missing, pred, sfreq=250))
        out_rows.append(
            {
                "test_subject": test_subject,
                "n_input_channels": len(input_channels),
                "input_channels": ",".join(input_channels),
                "n_reconstructed_channels": len(missing_channels),
                "method": method_name,
                **metrics,
                "n_test_trials": int(n_trials),
                "n_test_samples": int(n_trials * n_times),
            }
        )

    return out_rows


def main():
    parser = argparse.ArgumentParser(description="Phase 3 hybrid residual benchmark on BCI IV-2a (LOSO)")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a",
        help="Directory containing preprocessed subject folders and loso_splits.json",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=200000,
        help="Cap train sample rows for speed; set <=0 to disable cap",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-splits",
        type=int,
        default=0,
        help="Limit number of LOSO splits (0 means all)",
    )
    parser.add_argument(
        "--residual-model",
        type=str,
        default="ridge",
        choices=["ridge", "mlp"],
        help="Regressor used for residual prediction",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase3_hybrid_loso.csv",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase3_hybrid_summary.json",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir.resolve()
    loso_path = processed_dir / "loso_splits.json"
    if not loso_path.exists():
        raise FileNotFoundError(f"Missing LOSO split file: {loso_path}")

    loso_splits = json.loads(loso_path.read_text(encoding="utf-8"))
    if args.max_splits > 0:
        loso_splits = loso_splits[: args.max_splits]

    channel_sets = [CHANNEL_SET_5, CHANNEL_SET_10, CHANNEL_SET_15]
    rng = np.random.default_rng(args.seed)

    all_rows = []
    for split in loso_splits:
        test_subject = split["test_subject"]
        train_subjects = split["train_subjects"]
        for ch_set in channel_sets:
            rows = run_split(
                train_subjects=train_subjects,
                test_subject=test_subject,
                processed_dir=processed_dir,
                input_channels=ch_set,
                max_train_samples=args.max_train_samples,
                rng=rng,
                residual_model=args.residual_model,
            )
            all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("No Phase 3 rows generated")

    out_df = pd.DataFrame(all_rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_csv.resolve()
    out_df.to_csv(out_csv, index=False)

    metric_cols = [
        c
        for c in out_df.columns
        if c
        not in {
            "test_subject",
            "n_input_channels",
            "input_channels",
            "n_reconstructed_channels",
            "method",
            "n_test_trials",
            "n_test_samples",
        }
    ]
    summary_df = out_df.groupby(["n_input_channels", "method"], as_index=False)[metric_cols].mean().sort_values(
        ["n_input_channels", "method"]
    )

    summary_payload = {
        "rows": int(len(out_df)),
        "splits": int(len(loso_splits)),
        "residual_model": args.residual_model,
        "channel_sets": {
            "5": CHANNEL_SET_5,
            "10": CHANNEL_SET_10,
            "15": CHANNEL_SET_15,
        },
        "mean_metrics": summary_df.to_dict(orient="records"),
    }

    out_json = args.output_summary_json.resolve()
    out_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Saved detailed Phase 3 results: {out_csv}")
    print(f"Saved Phase 3 summary: {out_json}")


if __name__ == "__main__":
    main()
