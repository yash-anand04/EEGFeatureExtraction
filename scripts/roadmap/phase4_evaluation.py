import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, required: list[str], name: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {name}: {missing}")


def _numeric_metric_columns(df: pd.DataFrame):
    exclude = {
        "test_subject",
        "n_input_channels",
        "input_channels",
        "n_reconstructed_channels",
        "method",
        "n_test_trials",
        "n_test_samples",
    }
    metric_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            metric_cols.append(c)
    return metric_cols


def main():
    parser = argparse.ArgumentParser(description="Phase 4 evaluation aggregation for EEG reconstruction roadmap")
    parser.add_argument(
        "--phase2-csv",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase2_loso_fastbaselines.csv",
    )
    parser.add_argument(
        "--phase3-csv",
        type=Path,
        default=None,
        help="Single Phase 3 CSV path (kept for compatibility)",
    )
    parser.add_argument(
        "--phase3-csvs",
        type=str,
        default="",
        help="Comma-separated Phase 3 CSV paths to merge (e.g., ridge + mlp)",
    )
    parser.add_argument(
        "--extra-csvs",
        type=str,
        default="",
        help="Comma-separated additional benchmark CSVs to include (e.g., BEM/deep Phase 2 outputs)",
    )
    parser.add_argument(
        "--output-merged-csv",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase4_all_methods.csv",
    )
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase4_summary_by_method.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase4_evaluation_summary.json",
    )
    args = parser.parse_args()

    df_p2 = pd.read_csv(args.phase2_csv)

    phase3_paths = []
    if args.phase3_csvs:
        phase3_paths.extend([Path(p.strip()) for p in args.phase3_csvs.split(",") if p.strip()])
    elif args.phase3_csv is not None:
        phase3_paths.append(args.phase3_csv)
    else:
        phase3_paths.append(Path("processed") / "bci_competition_iv_2a" / "phase3_hybrid_loso.csv")

    df_p3_list = [pd.read_csv(p) for p in phase3_paths]
    df_p3 = pd.concat(df_p3_list, ignore_index=True)

    extra_paths = [Path(p.strip()) for p in args.extra_csvs.split(",") if p.strip()]
    extra_dfs = [pd.read_csv(p) for p in extra_paths]

    required = ["test_subject", "n_input_channels", "method", "rmse", "mae", "pearson_r", "r2"]
    _require_columns(df_p2, required, "phase2")
    _require_columns(df_p3, required, "phase3")

    df_all_parts = [df_p2, df_p3]
    df_all_parts.extend(extra_dfs)
    df_all = pd.concat(df_all_parts, ignore_index=True)
    metric_cols = _numeric_metric_columns(df_all)

    summary = (
        df_all.groupby(["n_input_channels", "method"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        col if isinstance(col, str) else (col[0] if col[1] == "" else f"{col[0]}_{col[1]}")
        for col in summary.columns
    ]

    rmse_rank = (
        df_all.groupby("method", as_index=False)["rmse"].mean().sort_values("rmse", ascending=True)
    )

    scale_rows = []
    for method_name, group in df_all.groupby("method"):
        grp = group.groupby("n_input_channels", as_index=False)["rmse"].mean().sort_values("n_input_channels")
        if len(grp) >= 2:
            slope = np.polyfit(grp["n_input_channels"].to_numpy(), grp["rmse"].to_numpy(), 1)[0]
        else:
            slope = np.nan
        scale_rows.append({"method": method_name, "rmse_vs_channels_slope": float(slope)})
    scaling_df = pd.DataFrame(scale_rows).sort_values("rmse_vs_channels_slope")

    best_by_input = (
        df_all.groupby(["n_input_channels", "method"], as_index=False)["rmse"].mean()
        .sort_values(["n_input_channels", "rmse"], ascending=[True, True])
        .groupby("n_input_channels", as_index=False)
        .first()
    )

    args.output_merged_csv.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(args.output_merged_csv, index=False)
    summary.to_csv(args.output_summary_csv, index=False)

    payload = {
        "n_rows_phase2": int(len(df_p2)),
        "n_rows_phase3": int(len(df_p3)),
        "n_rows_extra": int(sum(len(df) for df in extra_dfs)),
        "n_rows_total": int(len(df_all)),
        "overall_rmse_ranking": rmse_rank.to_dict(orient="records"),
        "best_method_by_input_channels": best_by_input.to_dict(orient="records"),
        "scaling_slope_rmse_vs_channels": scaling_df.to_dict(orient="records"),
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved merged metrics: {args.output_merged_csv.resolve()}")
    print(f"Saved method summary: {args.output_summary_csv.resolve()}")
    print(f"Saved Phase 4 evaluation JSON: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
