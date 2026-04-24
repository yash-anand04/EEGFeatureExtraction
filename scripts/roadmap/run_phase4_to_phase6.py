import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cmd(cmd: list[str]):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run roadmap Phase 4 -> Phase 6 scripts")
    parser.add_argument("--python", type=str, default=sys.executable)
    args = parser.parse_args()

    py = args.python

    run_cmd([py, str(REPO_ROOT / "scripts" / "roadmap" / "phase4_evaluation.py")])
    run_cmd([py, str(REPO_ROOT / "scripts" / "roadmap" / "phase4_spatial_ablation.py")])
    run_cmd([py, str(REPO_ROOT / "scripts" / "roadmap" / "phase5_visualizations.py")])
    run_cmd([py, str(REPO_ROOT / "scripts" / "roadmap" / "phase6_analysis_report.py")])

    print("Completed roadmap phases 4-6.")


if __name__ == "__main__":
    main()
