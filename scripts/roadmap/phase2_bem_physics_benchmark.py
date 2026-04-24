import argparse
import json
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.minimum_norm import apply_inverse, make_inverse_operator

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
    load_subject_data,
)


def _build_info(ch_names, sfreq):
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(montage, on_missing="ignore")
    raw_tmp = mne.io.RawArray(np.zeros((len(ch_names), 2), dtype=np.float64), info, verbose=False)
    raw_tmp.set_eeg_reference("average", projection=True, verbose=False)
    return raw_tmp.info


def _build_forward_models(sfreq: int, spacing: str = "oct4"):
    fs_dir = Path(mne.datasets.fetch_fsaverage(verbose=False))
    subjects_dir = fs_dir.parent
    trans_file = fs_dir / "bem" / "fsaverage-trans.fif"

    src = mne.setup_source_space("fsaverage", spacing=spacing, subjects_dir=subjects_dir, add_dist=False)
    model = mne.make_bem_model(
        subject="fsaverage",
        ico=3,
        conductivity=(0.33, 0.0132, 0.33),
        subjects_dir=subjects_dir,
    )
    bem = mne.make_bem_solution(model)

    info_full = _build_info(CHANNELS_22, sfreq)
    fwd_full = mne.make_forward_solution(
        info=info_full,
        trans=str(trans_file),
        src=src,
        bem=bem,
        meg=False,
        eeg=True,
        mindist=5.0,
        n_jobs=1,
        verbose=False,
    )
    return info_full, fwd_full


def _predict_trial_bem(full_trial_ch_time, input_channels, missing_channels, info_full, fwd_full, inv_cache, sfreq):
    input_idx = [CHANNELS_22.index(c) for c in input_channels]
    missing_idx = [CHANNELS_22.index(c) for c in missing_channels]

    info_keep = _build_info(input_channels, sfreq)
    cache_key = tuple(input_channels)
    if cache_key not in inv_cache:
        fwd_keep = mne.pick_channels_forward(fwd_full, include=input_channels, ordered=True)
        cov = mne.make_ad_hoc_cov(info_keep, std=1e-6)
        inv = make_inverse_operator(info_keep, fwd_keep, cov, loose=0.2, depth=0.8, verbose=False)
        inv_cache[cache_key] = inv
    inv = inv_cache[cache_key]

    # MNE inverse/forward expects EEG data in Volts.
    data_keep = full_trial_ch_time[input_idx, :] * 1e-6
    evoked_keep = mne.EvokedArray(data_keep, info_keep, tmin=0.0, verbose=False)

    stc = apply_inverse(evoked_keep, inv, lambda2=1.0, method="MNE", pick_ori=None, verbose=False)
    pred_evoked_full = mne.apply_forward(fwd_full, stc, info_full, verbose=False)

    # Convert predictions back to original dataset scale (microvolts-like units).
    y_true_missing = full_trial_ch_time[missing_idx, :].T
    y_pred_missing = pred_evoked_full.data[missing_idx, :].T * 1e6
    return y_true_missing, y_pred_missing


def run_split(test_subject, processed_dir: Path, input_channels, info_full, fwd_full, sfreq, max_trials_per_subject):
    missing_channels = [c for c in CHANNELS_22 if c not in input_channels]
    x_trials, _ = load_subject_data(processed_dir / test_subject)

    if max_trials_per_subject > 0 and x_trials.shape[0] > max_trials_per_subject:
        x_trials = x_trials[:max_trials_per_subject]

    inv_cache = {}
    y_true_blocks = []
    y_pred_blocks = []
    for trial_i in range(x_trials.shape[0]):
        y_true, y_pred = _predict_trial_bem(
            full_trial_ch_time=x_trials[trial_i],
            input_channels=input_channels,
            missing_channels=missing_channels,
            info_full=info_full,
            fwd_full=fwd_full,
            inv_cache=inv_cache,
            sfreq=sfreq,
        )
        y_true_blocks.append(y_true)
        y_pred_blocks.append(y_pred)

    y_true_all = np.concatenate(y_true_blocks, axis=0)
    y_pred_all = np.concatenate(y_pred_blocks, axis=0)

    metrics = compute_metrics(y_true_all, y_pred_all)
    metrics.update(compute_advanced_metrics(y_true_all, y_pred_all, sfreq=sfreq))

    n_times = x_trials.shape[2]
    return {
        "test_subject": test_subject,
        "n_input_channels": len(input_channels),
        "input_channels": ",".join(input_channels),
        "n_reconstructed_channels": len(missing_channels),
        "method": "bem_mne_forward_inverse",
        **metrics,
        "n_test_trials": int(x_trials.shape[0]),
        "n_test_samples": int(x_trials.shape[0] * n_times),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 2 physics-based BEM forward/inverse benchmark")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed") / "bci_competition_iv_2a")
    parser.add_argument("--sfreq", type=int, default=250)
    parser.add_argument("--max-splits", type=int, default=0)
    parser.add_argument(
        "--max-trials-per-subject",
        type=int,
        default=40,
        help="Cap trials per test subject for runtime control; <=0 means all",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase2_bem_physics_loso.csv",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase2_bem_physics_summary.json",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir.resolve()
    loso_splits = json.loads((processed_dir / "loso_splits.json").read_text(encoding="utf-8"))
    if args.max_splits > 0:
        loso_splits = loso_splits[: args.max_splits]

    info_full, fwd_full = _build_forward_models(sfreq=args.sfreq)
    channel_sets = [CHANNEL_SET_5, CHANNEL_SET_10, CHANNEL_SET_15]

    rows = []
    for split in loso_splits:
        test_subject = split["test_subject"]
        for ch_set in channel_sets:
            row = run_split(
                test_subject=test_subject,
                processed_dir=processed_dir,
                input_channels=ch_set,
                info_full=info_full,
                fwd_full=fwd_full,
                sfreq=args.sfreq,
                max_trials_per_subject=args.max_trials_per_subject,
            )
            rows.append(row)

    out_df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

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
    summary = out_df.groupby(["n_input_channels", "method"], as_index=False)[metric_cols].mean()

    payload = {
        "rows": int(len(out_df)),
        "splits": int(len(loso_splits)),
        "method": "bem_mne_forward_inverse",
        "mean_metrics": summary.to_dict(orient="records"),
    }
    args.output_summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved Phase 2 BEM results: {args.output_csv.resolve()}")
    print(f"Saved Phase 2 BEM summary: {args.output_summary_json.resolve()}")


if __name__ == "__main__":
    main()
