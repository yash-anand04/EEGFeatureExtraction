import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run full roadmap Phase 2 stack")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--processed-dir", type=Path, default=Path("processed") / "bci_competition_iv_2a")
    parser.add_argument("--max-splits", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=60000)
    parser.add_argument("--max-trials-per-subject", type=int, default=40)
    parser.add_argument("--deep-epochs", type=int, default=12)
    parser.add_argument("--deep-batch-size", type=int, default=512)
    parser.add_argument("--deep-max-test-samples", type=int, default=20000)
    args = parser.parse_args()

    py = args.python
    processed_dir = str(args.processed_dir)

    run_cmd(
        [
            py,
            str(REPO_ROOT / "scripts" / "channel_analysis" / "phase2_bci_baselines.py"),
            "--processed-dir",
            processed_dir,
            "--max-splits",
            str(args.max_splits),
            "--max-train-samples",
            str(args.max_train_samples),
            "--methods",
            "distance_weighted_interpolation,spherical_spline_interpolation,ridge_regression,mlp_regressor",
            "--output-csv",
            str(REPO_ROOT / "processed" / "bci_competition_iv_2a" / "phase2_loso_full_with_spline.csv"),
            "--output-summary-json",
            str(REPO_ROOT / "processed" / "bci_competition_iv_2a" / "phase2_loso_full_with_spline_summary.json"),
        ]
    )

    run_cmd(
        [
            py,
            str(REPO_ROOT / "scripts" / "roadmap" / "phase2_bem_physics_benchmark.py"),
            "--processed-dir",
            processed_dir,
            "--max-splits",
            str(args.max_splits),
            "--max-trials-per-subject",
            str(args.max_trials_per_subject),
            "--output-csv",
            str(REPO_ROOT / "processed" / "bci_competition_iv_2a" / "phase2_bem_physics_loso.csv"),
            "--output-summary-json",
            str(REPO_ROOT / "processed" / "bci_competition_iv_2a" / "phase2_bem_physics_summary.json"),
        ]
    )

    run_cmd(
        [
            py,
            str(REPO_ROOT / "scripts" / "roadmap" / "phase2_deep_models_benchmark.py"),
            "--processed-dir",
            processed_dir,
            "--max-splits",
            str(args.max_splits),
            "--max-train-samples",
            str(args.max_train_samples),
            "--max-test-samples",
            str(args.deep_max_test_samples),
            "--epochs",
            str(args.deep_epochs),
            "--batch-size",
            str(args.deep_batch_size),
            "--output-csv",
            str(REPO_ROOT / "processed" / "bci_competition_iv_2a" / "phase2_deep_models_loso.csv"),
            "--output-summary-json",
            str(REPO_ROOT / "processed" / "bci_competition_iv_2a" / "phase2_deep_models_summary.json"),
        ]
    )

    print("Completed full roadmap Phase 2 execution.")


if __name__ == "__main__":
    main()
