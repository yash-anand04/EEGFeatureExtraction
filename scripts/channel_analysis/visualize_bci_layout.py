from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from phase2_bci_baselines import CHANNEL_POS_2D, CHANNEL_SET_5, CHANNEL_SET_10, CHANNEL_SET_15


def plot_layout(output_path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), constrained_layout=True)
    fig.patch.set_facecolor("black")

    all_channels = list(CHANNEL_POS_2D.keys())
    sets = [
        ("All 22 EEG Channels", all_channels),
        ("Input Set (5)", CHANNEL_SET_5),
        ("Input Set (10)", CHANNEL_SET_10),
        ("Input Set (15)", CHANNEL_SET_15),
    ]

    for ax, (title, active_set) in zip(axes, sets):
        ax.set_facecolor("black")
        # Scalp outline and reference marks.
        scalp = Circle((0, 0), 1.0, edgecolor="#f8fafc", facecolor="none", lw=1.5)
        ax.add_patch(scalp)
        ax.plot([0, -0.08, 0.08, 0], [1.0, 1.08, 1.08, 1.0], color="#f8fafc", lw=1.2)  # nose
        ax.plot([-1.0, -1.06], [0.0, 0.0], color="#f8fafc", lw=1.2)  # left ear marker
        ax.plot([1.0, 1.06], [0.0, 0.0], color="#f8fafc", lw=1.2)   # right ear marker

        for ch, (x, y) in CHANNEL_POS_2D.items():
            if ch in active_set:
                color = "#2563eb"
                size = 55
                alpha = 1.0
            else:
                color = "#e2e8f0"
                size = 30
                alpha = 0.7
            ax.scatter(x, y, c=color, s=size, alpha=alpha, zorder=3, edgecolors="white", linewidths=0.5)
            ax.text(x + 0.015, y + 0.015, ch, fontsize=8, color="#f8fafc")

        ax.set_title(title, fontsize=11, color="#f8fafc")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.1, 1.2)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    fig.suptitle("BCI IV-2a Electrode Layout Used in Phase 2 Baselines", fontsize=13, color="#f8fafc")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    out = root / "docs" / "bci_iv2a_electrode_layout.png"
    plot_layout(out)
    print(f"Saved: {out}")
