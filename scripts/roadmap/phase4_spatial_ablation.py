import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.channel_analysis.phase2_bci_baselines import (  # noqa: E402
    CHANNELS_22,
    compute_metrics,
    distance_weighted_interpolation,
    load_subject_data,
    to_sample_matrix,
)


def evaluate_removed_channel(processed_dir: Path, removed_channel: str, max_samples_per_subject: int, rng: np.random.Generator):
    input_channels = [c for c in CHANNELS_22 if c != removed_channel]
    target_channels = [removed_channel]
    target_idx = [CHANNELS_22.index(removed_channel)]
    input_idx = [CHANNELS_22.index(c) for c in input_channels]

    subject_dirs = sorted([p for p in processed_dir.iterdir() if p.is_dir() and p.name.startswith("A")])

    all_true = []
    all_pred = []
    for subject_dir in subject_dirs:
        x_trials, _ = load_subject_data(subject_dir)
        samples = to_sample_matrix(x_trials)

        if max_samples_per_subject > 0 and samples.shape[0] > max_samples_per_subject:
            sel = rng.choice(samples.shape[0], size=max_samples_per_subject, replace=False)
            samples = samples[sel]

        x_in = samples[:, input_idx]
        y_true = samples[:, target_idx]
        y_pred = distance_weighted_interpolation(x_in, input_channels, target_channels)

        all_true.append(y_true)
        all_pred.append(y_pred)

    y_true_all = np.concatenate(all_true, axis=0)
    y_pred_all = np.concatenate(all_pred, axis=0)
    metrics = compute_metrics(y_true_all, y_pred_all)
    metrics["n_samples"] = int(y_true_all.shape[0])
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 4 spatial ablation: remove one electrode at a time")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a",
    )
    parser.add_argument(
        "--max-samples-per-subject",
        type=int,
        default=12000,
        help="Sample cap per subject for runtime control; <=0 means all samples",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase4_spatial_ablation.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase4_spatial_ablation_summary.json",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for removed in CHANNELS_22:
        metrics = evaluate_removed_channel(
            processed_dir=args.processed_dir.resolve(),
            removed_channel=removed,
            max_samples_per_subject=args.max_samples_per_subject,
            rng=rng,
        )
        rows.append({"removed_channel": removed, **metrics})

    df = pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    summary = {
        "top_5_most_critical": df.head(5).to_dict(orient="records"),
        "least_critical": df.tail(5).to_dict(orient="records"),
    }
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved spatial ablation CSV: {args.output_csv.resolve()}")
    print(f"Saved spatial ablation JSON: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
