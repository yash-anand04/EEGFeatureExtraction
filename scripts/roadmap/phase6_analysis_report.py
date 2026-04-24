import argparse
import json
from pathlib import Path

import pandas as pd


def _format_table(df: pd.DataFrame, columns: list[str], max_rows: int = 10):
    sub = df[columns].head(max_rows).copy()
    return sub.to_markdown(index=False)


def main():
    parser = argparse.ArgumentParser(description="Phase 6 report synthesis from Phase 4/5 artifacts")
    parser.add_argument(
        "--phase4-json",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase4_evaluation_summary.json",
    )
    parser.add_argument(
        "--phase4-all-csv",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase4_all_methods.csv",
    )
    parser.add_argument(
        "--ablation-csv",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase4_spatial_ablation.csv",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("Report") / "phase6_analysis_report.md",
    )
    parser.add_argument(
        "--latency-json",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase6_latency_benchmark_summary.json",
    )
    parser.add_argument(
        "--phase5-publication-dir",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase5_publication_plots",
    )
    args = parser.parse_args()

    summary = json.loads(args.phase4_json.read_text(encoding="utf-8"))
    df_all = pd.read_csv(args.phase4_all_csv)
    df_ablation = pd.read_csv(args.ablation_csv)

    rmse_rank_df = pd.DataFrame(summary.get("overall_rmse_ranking", []))
    best_by_input_df = pd.DataFrame(summary.get("best_method_by_input_channels", []))

    band_cols = [
        "delta_band_rmse",
        "theta_band_rmse",
        "alpha_band_rmse",
        "beta_band_rmse",
        "gamma_band_rmse",
    ]
    existing_band_cols = [c for c in band_cols if c in df_all.columns]

    band_text = "No frequency-band metrics available."
    if existing_band_cols:
        band_mean = df_all[existing_band_cols].mean().sort_values()
        strongest = band_mean.index[0].replace("_band_rmse", "")
        weakest = band_mean.index[-1].replace("_band_rmse", "")
        band_text = f"Best-preserved band (lowest mean RMSE): {strongest}; Most difficult band (highest mean RMSE): {weakest}"

    critical = df_ablation.sort_values("rmse", ascending=False).head(5)

    latency_df = pd.DataFrame()
    if args.latency_json.exists():
        latency_payload = json.loads(args.latency_json.read_text(encoding="utf-8"))
        latency_df = pd.DataFrame(latency_payload.get("mean_latency", []))

    threshold_rmse = 2.5
    channel_choice_text = "No channel count reached RMSE <= 2.5 with current methods."
    if not best_by_input_df.empty and "rmse" in best_by_input_df.columns:
        hit = best_by_input_df[best_by_input_df["rmse"] <= threshold_rmse].sort_values("n_input_channels")
        if not hit.empty:
            row = hit.iloc[0]
            channel_choice_text = (
                f"Minimum channel count meeting RMSE <= {threshold_rmse}: "
                f"{int(row['n_input_channels'])} using {row['method']} (RMSE={row['rmse']:.4f})"
            )

    lines = []
    lines.append("# Phase 6 Analysis Report")
    lines.append("")
    lines.append("## Benchmark Cross-Comparison")
    lines.append("")
    if rmse_rank_df.empty:
        lines.append("No RMSE ranking data available.")
    else:
        lines.append(_format_table(rmse_rank_df, ["method", "rmse"], max_rows=20))

    lines.append("")
    lines.append("## Best Method Per Channel Count")
    lines.append("")
    if best_by_input_df.empty:
        lines.append("No best-by-channel data available.")
    else:
        lines.append(_format_table(best_by_input_df, ["n_input_channels", "method", "rmse"], max_rows=10))

    lines.append("")
    lines.append("## Insights Generation")
    lines.append("")
    lines.append(f"- {channel_choice_text}")
    lines.append(f"- {band_text}")
    lines.append("- Most critical missing electrodes from global ablation:")
    lines.append(critical[["removed_channel", "rmse", "pearson_r"]].to_markdown(index=False))

    lines.append("")
    lines.append("## Latency Analysis")
    lines.append("")
    if latency_df.empty:
        lines.append("No latency benchmark summary found.")
    else:
        lines.append(_format_table(latency_df, ["method", "train_seconds", "predict_seconds", "predict_ms_per_sample"], max_rows=20))

    lines.append("")
    lines.append("## Practical Conclusions")
    lines.append("")
    lines.append("- Hybrid residual models can be directly compared against interpolation baselines with unified signal, spatial, and spectral metrics.")
    lines.append("- Electrode ablation exposes location-specific failure points that should guide wearable channel placement design.")
    lines.append("- Latency trends are now included to support real-time feasibility discussion.")

    lines.append("")
    lines.append("## Publication Visual Artifacts")
    lines.append("")
    if args.phase5_publication_dir.exists():
        for name in [
            "timeseries_overlay.png",
            "topomap_error_maps.png",
            "psd_comparison.png",
            "electrode_error_topology.png",
        ]:
            p = args.phase5_publication_dir / name
            if p.exists():
                lines.append(f"- {p.as_posix()}")
    else:
        lines.append("No Phase 5 publication plot directory found.")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved Phase 6 report: {args.output_md.resolve()}")


if __name__ == "__main__":
    main()
