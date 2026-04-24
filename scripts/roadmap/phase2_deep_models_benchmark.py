import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.channel_analysis.phase2_bci_baselines import (  # noqa: E402
    CHANNELS_22,
    CHANNEL_SET_5,
    CHANNEL_SET_10,
    CHANNEL_SET_15,
    compute_advanced_metrics,
    compute_metrics,
    load_subject_data,
    to_sample_matrix,
)


class DeepAutoencoderRegressor(nn.Module):
    def __init__(self, n_input: int, n_output: int):
        super().__init__()
        latent = max(16, n_input * 2)
        self.encoder = nn.Sequential(
            nn.Linear(n_input, 128),
            nn.ReLU(),
            nn.Linear(128, latent),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 128),
            nn.ReLU(),
            nn.Linear(128, n_output),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class CNNRegressor(nn.Module):
    def __init__(self, n_input: int, n_output: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * n_input, 128),
            nn.ReLU(),
            nn.Linear(128, n_output),
        )

    def forward(self, x):
        # x: (batch, n_input)
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        return self.fc(x)


def _train_model(model, x_train, y_train, epochs: int, batch_size: int, lr: float, device: str):
    model = model.to(device)
    x_t = torch.tensor(x_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)
    ds = TensorDataset(x_t, y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    return model


def _predict_model(model, x, device: str):
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        pred = model(x_t).cpu().numpy()
    return pred


def run_split(train_subjects, test_subject, processed_dir: Path, input_channels, max_train_samples: int, max_test_samples: int, seed: int, epochs: int, batch_size: int, lr: float, device: str):
    rng = np.random.default_rng(seed)

    input_idx = [CHANNELS_22.index(c) for c in input_channels]
    missing_channels = [c for c in CHANNELS_22 if c not in input_channels]
    missing_idx = [CHANNELS_22.index(c) for c in missing_channels]

    train_blocks = []
    for subj in train_subjects:
        x_trials, _ = load_subject_data(processed_dir / subj)
        train_blocks.append(to_sample_matrix(x_trials))
    train_samples = np.concatenate(train_blocks, axis=0)

    if max_train_samples > 0 and train_samples.shape[0] > max_train_samples:
        sel = rng.choice(train_samples.shape[0], size=max_train_samples, replace=False)
        train_samples = train_samples[sel]

    x_test_trials, _ = load_subject_data(processed_dir / test_subject)
    n_trials, _, n_times = x_test_trials.shape
    test_samples = to_sample_matrix(x_test_trials)
    if max_test_samples > 0 and test_samples.shape[0] > max_test_samples:
        sel = rng.choice(test_samples.shape[0], size=max_test_samples, replace=False)
        test_samples = test_samples[sel]

    x_train = train_samples[:, input_idx]
    y_train = train_samples[:, missing_idx]
    x_test = test_samples[:, input_idx]
    y_test = test_samples[:, missing_idx]

    ae = DeepAutoencoderRegressor(n_input=x_train.shape[1], n_output=y_train.shape[1])
    ae = _train_model(ae, x_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr, device=device)
    pred_ae = _predict_model(ae, x_test, device=device)

    cnn = CNNRegressor(n_input=x_train.shape[1], n_output=y_train.shape[1])
    cnn = _train_model(cnn, x_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr, device=device)
    pred_cnn = _predict_model(cnn, x_test, device=device)

    rows = []
    for method, pred in [
        ("deep_autoencoder", pred_ae),
        ("deep_cnn", pred_cnn),
    ]:
        metrics = compute_metrics(y_test, pred)
        metrics.update(compute_advanced_metrics(y_test, pred, sfreq=250))
        rows.append(
            {
                "test_subject": test_subject,
                "n_input_channels": len(input_channels),
                "input_channels": ",".join(input_channels),
                "n_reconstructed_channels": len(missing_channels),
                "method": method,
                **metrics,
                "n_test_trials": int(n_trials),
                "n_test_samples": int(test_samples.shape[0]),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Phase 2 deep models (Autoencoder + CNN) LOSO benchmark")
    parser.add_argument("--processed-dir", type=Path, default=Path("processed") / "bci_competition_iv_2a")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-splits", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=60000)
    parser.add_argument("--max-test-samples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase2_deep_models_loso.csv",
    )
    parser.add_argument(
        "--output-summary-json",
        type=Path,
        default=Path("processed") / "bci_competition_iv_2a" / "phase2_deep_models_summary.json",
    )
    args = parser.parse_args()

    processed_dir = args.processed_dir.resolve()
    loso_splits = json.loads((processed_dir / "loso_splits.json").read_text(encoding="utf-8"))
    if args.max_splits > 0:
        loso_splits = loso_splits[: args.max_splits]

    channel_sets = [CHANNEL_SET_5, CHANNEL_SET_10, CHANNEL_SET_15]

    rows = []
    for split in loso_splits:
        test_subject = split["test_subject"]
        train_subjects = split["train_subjects"]
        for ch_set in channel_sets:
            split_rows = run_split(
                train_subjects=train_subjects,
                test_subject=test_subject,
                processed_dir=processed_dir,
                input_channels=ch_set,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
                seed=args.seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )
            rows.extend(split_rows)

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
        "epochs": int(args.epochs),
        "device": args.device,
        "methods": ["deep_autoencoder", "deep_cnn"],
        "mean_metrics": summary.to_dict(orient="records"),
    }
    args.output_summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved Phase 2 deep model results: {args.output_csv.resolve()}")
    print(f"Saved Phase 2 deep model summary: {args.output_summary_json.resolve()}")


if __name__ == "__main__":
    main()
