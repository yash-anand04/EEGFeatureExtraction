import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.channel_analysis.phase2_bci_baselines import (  # noqa: E402
    CHANNELS_22,
    CHANNEL_SET_5,
    CHANNEL_SET_10,
    CHANNEL_SET_15,
    distance_weighted_interpolation,
    load_subject_data,
    to_sample_matrix,
)


def _prepare_split(processed_dir: Path, max_train_samples: int, max_test_samples: int, seed: int):
    loso = json.loads((processed_dir / "loso_splits.json").read_text(encoding="utf-8"))
    split = loso[0]
    train_subjects = split["train_subjects"]
    test_subject = split["test_subject"]

    rng = np.random.default_rng(seed)

    train_blocks = []
    for subj in train_subjects:
        x_trials, _ = load_subject_data(processed_dir / subj)
        train_blocks.append(to_sample_matrix(x_trials))
    train_samples = np.concatenate(train_blocks, axis=0)

    if max_train_samples > 0 and train_samples.shape[0] > max_train_samples:
        sel = rng.choice(train_samples.shape[0], size=max_train_samples, replace=False)
        train_samples = train_samples[sel]

    x_test_trials, _ = load_subject_data(processed_dir / test_subject)
    test_samples = to_sample_matrix(x_test_trials)
    if max_test_samples > 0 and test_samples.shape[0] > max_test_samples:
        sel = rng.choice(test_samples.shape[0], size=max_test_samples, replace=False)
        test_samples = test_samples[sel]

    return train_samples, test_samples, test_subject


def _time_it(fn):
    t0 = time.perf_counter()
    out = fn()
    t1 = time.perf_counter()
    return out, (t1 - t0)


def main():
    parser = argparse.ArgumentParser(description="Phase 6 latency benchmark for reconstruction methods")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a",
    )
    parser.add_argument("--max-train-samples", type=int, default=50000)
    parser.add_argument("--max-test-samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase6_latency_benchmark.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase6_latency_benchmark_summary.json",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir.resolve()
    train_samples, test_samples, test_subject = _prepare_split(
        processed_dir=processed_dir,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )

    rows = []
    for input_set in [CHANNEL_SET_5, CHANNEL_SET_10, CHANNEL_SET_15]:
        input_idx = [CHANNELS_22.index(c) for c in input_set]
        missing_channels = [c for c in CHANNELS_22 if c not in input_set]
        missing_idx = [CHANNELS_22.index(c) for c in missing_channels]

        x_train = train_samples[:, input_idx]
        y_train = train_samples[:, missing_idx]
        x_test = test_samples[:, input_idx]

        # Distance interpolation: no fit, predict-only timing.
        _, pred_t = _time_it(lambda: distance_weighted_interpolation(x_test, input_set, missing_channels))
        rows.append(
            {
                "method": "distance_weighted_interpolation",
                "n_input_channels": len(input_set),
                "train_seconds": 0.0,
                "predict_seconds": float(pred_t),
                "predict_ms_per_sample": float((pred_t * 1000.0) / max(1, x_test.shape[0])),
                "test_subject": test_subject,
                "n_test_samples": int(x_test.shape[0]),
            }
        )

        # Ridge baseline.
        ridge = Ridge(alpha=1.0)
        _, fit_t = _time_it(lambda: ridge.fit(x_train, y_train))
        _, pred_t = _time_it(lambda: ridge.predict(x_test))
        rows.append(
            {
                "method": "ridge_regression",
                "n_input_channels": len(input_set),
                "train_seconds": float(fit_t),
                "predict_seconds": float(pred_t),
                "predict_ms_per_sample": float((pred_t * 1000.0) / max(1, x_test.shape[0])),
                "test_subject": test_subject,
                "n_test_samples": int(x_test.shape[0]),
            }
        )

        # Pure AI MLP baseline.
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
        _, fit_t = _time_it(lambda: mlp.fit(x_train, y_train))
        _, pred_t = _time_it(lambda: mlp.predict(x_test))
        rows.append(
            {
                "method": "mlp_regression",
                "n_input_channels": len(input_set),
                "train_seconds": float(fit_t),
                "predict_seconds": float(pred_t),
                "predict_ms_per_sample": float((pred_t * 1000.0) / max(1, x_test.shape[0])),
                "test_subject": test_subject,
                "n_test_samples": int(x_test.shape[0]),
            }
        )

        # Hybrid ridge residual.
        y_interp_train = distance_weighted_interpolation(x_train, input_set, missing_channels)
        y_resid = y_train - y_interp_train
        x_h_train = np.concatenate([x_train, y_interp_train], axis=1)

        h_ridge = Ridge(alpha=1.0)
        _, fit_t = _time_it(lambda: h_ridge.fit(x_h_train, y_resid))
        y_interp_test = distance_weighted_interpolation(x_test, input_set, missing_channels)
        x_h_test = np.concatenate([x_test, y_interp_test], axis=1)
        _, pred_t = _time_it(lambda: h_ridge.predict(x_h_test))
        rows.append(
            {
                "method": "hybrid_residual_ridge",
                "n_input_channels": len(input_set),
                "train_seconds": float(fit_t),
                "predict_seconds": float(pred_t),
                "predict_ms_per_sample": float((pred_t * 1000.0) / max(1, x_test.shape[0])),
                "test_subject": test_subject,
                "n_test_samples": int(x_test.shape[0]),
            }
        )

    out_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    summary = (
        out_df.groupby("method", as_index=False)[["train_seconds", "predict_seconds", "predict_ms_per_sample"]]
        .mean()
        .sort_values("predict_ms_per_sample")
    )
    args.output_json.write_text(
        json.dumps({"mean_latency": summary.to_dict(orient="records")}, indent=2),
        encoding="utf-8",
    )

    print(f"Saved latency CSV: {args.output_csv.resolve()}")
    print(f"Saved latency summary JSON: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
