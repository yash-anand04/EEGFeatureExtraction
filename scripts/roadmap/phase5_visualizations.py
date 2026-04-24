import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_rmse_scaling(df_summary: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(10, 6))
    for method, grp in df_summary.groupby("method"):
        plt.plot(grp["n_input_channels"], grp["rmse_mean"], marker="o", linewidth=2, label=method)
    plt.xlabel("Input Channels")
    plt.ylabel("RMSE")
    plt.title("Channel Scaling: RMSE vs Input Channel Count")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_corr_scaling(df_summary: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(10, 6))
    if "pearson_r_mean" not in df_summary.columns:
        return
    for method, grp in df_summary.groupby("method"):
        plt.plot(grp["n_input_channels"], grp["pearson_r_mean"], marker="o", linewidth=2, label=method)
    plt.xlabel("Input Channels")
    plt.ylabel("Pearson r")
    plt.title("Channel Scaling: Correlation vs Input Channel Count")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_band_heatmap(df_all: pd.DataFrame, out_path: Path):
    band_cols = [
        "delta_band_rmse",
        "theta_band_rmse",
        "alpha_band_rmse",
        "beta_band_rmse",
        "gamma_band_rmse",
    ]
    existing = [c for c in band_cols if c in df_all.columns]
    if not existing:
        return

    band_df = df_all.groupby("method", as_index=False)[existing].mean().set_index("method")
    data = band_df.to_numpy()

    plt.figure(figsize=(10, 6))
    im = plt.imshow(data, aspect="auto", cmap="viridis")
    plt.yticks(np.arange(len(band_df.index)), band_df.index, fontsize=8)
    plt.xticks(np.arange(len(existing)), [c.replace("_band_rmse", "") for c in existing])
    plt.colorbar(im, label="Band RMSE")
    plt.title("Frequency Band Errors by Method")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_ablation(df_ablation: pd.DataFrame, out_path: Path):
    top = df_ablation.sort_values("rmse", ascending=False).head(12)
    plt.figure(figsize=(11, 6))
    plt.bar(top["removed_channel"], top["rmse"], color="#B03A2E")
    plt.ylabel("RMSE after removal")
    plt.xlabel("Removed channel")
    plt.title("Spatial Ablation: Most Critical Electrodes")
    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_publication_panel(df_summary: pd.DataFrame, df_ablation: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for method, grp in df_summary.groupby("method"):
        axes[0].plot(grp["n_input_channels"], grp["rmse_mean"], marker="o", linewidth=2, label=method)
    axes[0].set_title("RMSE Scaling")
    axes[0].set_xlabel("Input Channels")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.3)

    top = df_ablation.sort_values("rmse", ascending=False).head(8)
    axes[1].bar(top["removed_channel"], top["rmse"], color="#117A65")
    axes[1].set_title("Top Electrode Importance")
    axes[1].set_xlabel("Removed channel")
    axes[1].set_ylabel("RMSE")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 5 visualization generation")
    parser.add_argument(
        "--phase4-all-csv",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase4_all_methods.csv",
    )
    parser.add_argument(
        "--phase4-summary-csv",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase4_summary_by_method.csv",
    )
    parser.add_argument(
        "--ablation-csv",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase4_spatial_ablation.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase5_plots",
    )
    args = parser.parse_args()

    df_all = pd.read_csv(args.phase4_all_csv)
    df_summary = pd.read_csv(args.phase4_summary_csv)
    df_ablation = pd.read_csv(args.ablation_csv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _plot_rmse_scaling(df_summary, args.out_dir / "rmse_scaling.png")
    _plot_corr_scaling(df_summary, args.out_dir / "correlation_scaling.png")
    _plot_band_heatmap(df_all, args.out_dir / "band_rmse_heatmap.png")
    _plot_ablation(df_ablation, args.out_dir / "ablation_top_channels.png")
    _plot_publication_panel(df_summary, df_ablation, args.out_dir / "publication_core_panel.png")

    print(f"Saved Phase 5 plots to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
