import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

# BCI Competition IV 2a canonical EEG channel names (22 EEG + 3 EOG in raw file)
BCI_IV_2A_EEG_CHANNELS = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4",
    "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz"
]

CLASS_ID_TO_NAME = {
    1: "left_hand",
    2: "right_hand",
    3: "feet",
    4: "tongue",
}


def _iter_runs(mat_path: Path):
    """Yield run objects from a BCI .mat file."""
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)["data"]
    if not isinstance(data, np.ndarray):
        data = np.array([data], dtype=object)
    for run_idx, run in enumerate(data):
        yield run_idx, run


def _extract_trials_from_run(run, subject_id: str, run_idx: int, tmin_s: float, tmax_s: float, drop_artifacts: bool):
    """Extract fixed windows from one run into (trial, channel, time) arrays."""
    x = np.asarray(run.X, dtype=np.float32)  # shape: (samples, channels=25)
    y = np.asarray(run.y).reshape(-1).astype(np.int32)
    trial_onsets = np.asarray(run.trial).reshape(-1).astype(np.int64)
    artifacts = np.asarray(run.artifacts).reshape(-1).astype(np.int32)
    fs = int(run.fs)

    # Runs with empty y/trial are not MI task runs (typically calibration/EOG runs).
    if y.size == 0 or trial_onsets.size == 0:
        return None, None

    start_offset = int(round(tmin_s * fs))
    end_offset = int(round(tmax_s * fs))
    if end_offset <= start_offset:
        raise ValueError("tmax must be greater than tmin")

    trial_windows = []
    records = []

    for i, onset in enumerate(trial_onsets):
        label_id = int(y[i])
        artifact_flag = int(artifacts[i]) if i < artifacts.size else 0

        if drop_artifacts and artifact_flag == 1:
            continue

        start = int(onset + start_offset)
        end = int(onset + end_offset)

        # Skip truncated trials at boundaries.
        if start < 0 or end > x.shape[0]:
            continue

        # Keep only first 22 EEG channels; last 3 channels are EOG.
        window = x[start:end, :22].T  # shape: (22, time)
        trial_windows.append(window)

        records.append(
            {
                "subject": subject_id,
                "run_index": run_idx,
                "trial_index_in_run": i,
                "label_id": label_id,
                "label_name": CLASS_ID_TO_NAME.get(label_id, f"class_{label_id}"),
                "artifact": artifact_flag,
                "start_sample": start,
                "end_sample": end,
                "sampling_rate": fs,
            }
        )

    if not trial_windows:
        return None, None

    return np.stack(trial_windows, axis=0), pd.DataFrame.from_records(records)


def preprocess_file(mat_path: Path, output_dir: Path, tmin_s: float, tmax_s: float, drop_artifacts: bool):
    """Preprocess one BCI .mat file and write NPZ + metadata CSV."""
    subject_id = mat_path.stem

    all_x = []
    all_meta = []

    for run_idx, run in _iter_runs(mat_path):
        run_x, run_meta = _extract_trials_from_run(
            run,
            subject_id=subject_id,
            run_idx=run_idx,
            tmin_s=tmin_s,
            tmax_s=tmax_s,
            drop_artifacts=drop_artifacts,
        )
        if run_x is None:
            continue
        all_x.append(run_x)
        all_meta.append(run_meta)

    if not all_x:
        return None

    x_all = np.concatenate(all_x, axis=0).astype(np.float32)
    meta_df = pd.concat(all_meta, ignore_index=True)
    y_all = meta_df["label_id"].to_numpy(dtype=np.int32)

    subject_out = output_dir / subject_id
    subject_out.mkdir(parents=True, exist_ok=True)

    npz_path = subject_out / "trials.npz"
    csv_path = subject_out / "metadata.csv"

    np.savez_compressed(
        npz_path,
        X=x_all,
        y=y_all,
        channel_names=np.array(BCI_IV_2A_EEG_CHANNELS, dtype=object),
        class_names=np.array([CLASS_ID_TO_NAME[i] for i in sorted(CLASS_ID_TO_NAME.keys())], dtype=object),
    )
    meta_df.to_csv(csv_path, index=False)

    return {
        "subject": subject_id,
        "n_trials": int(x_all.shape[0]),
        "n_channels": int(x_all.shape[1]),
        "n_samples": int(x_all.shape[2]),
        "sampling_rate": int(meta_df["sampling_rate"].iloc[0]),
        "npz_path": str(npz_path),
        "csv_path": str(csv_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess BCI Competition IV 2a .mat files into train-ready tensors")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("Data_BCI_Competition"),
        help="Directory containing A0xT.mat files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a",
        help="Directory to store preprocessed outputs",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="A*T.mat",
        help="Glob pattern for input .mat files",
    )
    parser.add_argument(
        "--tmin",
        type=float,
        default=2.0,
        help="Window start in seconds relative to trial onset",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=6.0,
        help="Window end in seconds relative to trial onset",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep trials marked as artifact (default: drop them)",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(input_dir.glob(args.glob))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in {input_dir} with pattern {args.glob}")

    summaries = []
    for mat_path in mat_files:
        summary = preprocess_file(
            mat_path=mat_path,
            output_dir=output_dir,
            tmin_s=args.tmin,
            tmax_s=args.tmax,
            drop_artifacts=not args.keep_artifacts,
        )
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise RuntimeError("No trials were extracted. Check parsing window and input files.")

    summary_df = pd.DataFrame(summaries).sort_values("subject")
    summary_csv = output_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    subjects = summary_df["subject"].tolist()
    loso_splits = []
    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]
        loso_splits.append(
            {
                "test_subject": test_subject,
                "train_subjects": train_subjects,
            }
        )

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "glob": args.glob,
        "window_seconds": {"tmin": args.tmin, "tmax": args.tmax},
        "drop_artifacts": not args.keep_artifacts,
        "eeg_channels": BCI_IV_2A_EEG_CHANNELS,
        "classes": CLASS_ID_TO_NAME,
        "subjects_processed": summary_df["subject"].tolist(),
        "total_trials": int(summary_df["n_trials"].sum()),
        "loso_split_count": len(loso_splits),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    loso_path = output_dir / "loso_splits.json"
    loso_path.write_text(json.dumps(loso_splits, indent=2), encoding="utf-8")

    print(f"Processed {len(summary_df)} subjects")
    print(f"Total extracted trials: {manifest['total_trials']}")
    print(f"Summary written to: {summary_csv}")
    print(f"Manifest written to: {manifest_path}")
    print(f"LOSO splits written to: {loso_path}")


if __name__ == "__main__":
    main()
