# BCI IV-2a channel reconstruction (clean layout)

Self-contained Jupyter notebooks evaluate **Leave-One-Subject-Out** reconstruction from **5 / 10 / 15 → 22** EEG channels on BCI Competition IV-2a. Each method writes metrics under `Results/<Method>/`, plots under `Results_Visualization/<Method>/`, and trainable methods save artifacts under `Model_Files/`.

## Run order

1. **`Notebooks/00_preprocess_bci_iv2a.ipynb`** — place raw `.mat` files in `BCI_Competition_Data/`, then run to populate `Processed_BCI_Competition_Data/` plus `manifest.json` and `loso_splits.json`.
2. **Per-method notebooks** in `Notebooks/<Method>/` — run any or all (each notebook is standalone).
3. **`Notebooks/Comparison/comparison.ipynb`** — aggregates all `*_loso_metrics.csv` files into `Results/Comparison/` and figures in `Results_Visualization/Comparison/`.

## Layout

| Path | Role |
|------|------|
| `BCI_Competition_Data/` | Raw competition `.mat` inputs |
| `Processed_BCI_Competition_Data/` | Preprocessed `trials.npz` per subject and shared LOSO metadata |
| `Notebooks/` | Preprocessing, 11 methods, comparison |
| `Model_Files/` | Checkpoints, scalers, training plots (per method subfolders) |
| `Results/` | CSV + JSON summaries per method (`Comparison/` after aggregation) |
| `Results_Visualization/` | PNG figures per method (`Comparison/` after aggregation) |
