# Roadmap-Oriented Execution Scripts

This folder organizes the project workflow directly against `Report/Roadmap.md`.

## Phase Mapping

- Phase 1 (Dataset & Preprocessing)
  - `scripts/data_creation/preprocess_bci_competition.py`
  - `scripts/data_creation/validate_bci_preprocessing.py`
- Phase 2 (Baseline Models)
  - `scripts/channel_analysis/phase2_bci_baselines.py`
  - Physics and Pure-AI benchmark execution now lives in method-specific notebooks under `Main_codes/Baseline_analysis/`
- Phase 3 (Hybrid Model)
  - `scripts/hybrid_ai_approach/phase3_bci_hybrid_residual.py`
- Phase 4 (Evaluation)
  - `scripts/roadmap/phase4_evaluation.py`
  - `scripts/roadmap/phase4_spatial_ablation.py`
- Phase 5 (Visualization)
  - `scripts/roadmap/phase5_visualizations.py`
  - `scripts/roadmap/phase5_publication_visuals.py`
- Phase 6 (Analysis & Reporting)
  - `scripts/roadmap/phase6_analysis_report.py`
  - `scripts/roadmap/phase6_latency_benchmark.py`

## Recommended Order

1. Run Phase 2 full LOSO baselines.
2. Run Phase 3 full LOSO hybrid residual benchmarks.
3. Run Phase 4 evaluation aggregation and spatial ablation.
4. Run Phase 5 visual generation.
5. Run Phase 6 markdown report generation.
