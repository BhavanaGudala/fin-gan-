import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os

# Features that are all-zero / constant across the CICDDoS2019 dataset.
# These contribute only noise to reconstruction error.
DROP_FEATURES = [
    "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Count", "PSH Flag Count", "ECE Flag Count",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
]

# Heavy-tailed features (skew > 10) that benefit from log1p transform.
# Raw z-score normalization is ineffective on these distributions.
LOG_FEATURES = [
    "Fwd Act Data Packets", "Fwd Packets Length Total", "Subflow Fwd Bytes",
    "Total Backward Packets", "Subflow Bwd Packets", "Total Fwd Packets",
    "Subflow Fwd Packets", "Subflow Bwd Bytes", "Bwd Packets Length Total",
    "Flow IAT Min", "Fwd IAT Min", "Packet Length Variance",
    "Flow Duration", "Fwd IAT Total", "Flow IAT Max", "Fwd IAT Max",
    "Fwd IAT Mean", "Flow IAT Mean", "Flow IAT Std", "Fwd IAT Std",
    "Idle Max", "Idle Std", "Idle Mean", "Bwd IAT Mean", "Bwd IAT Std",
    "Bwd IAT Max", "Bwd IAT Total",
    "Fwd Packet Length Max", "Bwd Packet Length Max",
    "Packet Length Mean", "Avg Packet Size",
]


class FlowDataset(Dataset):
    def __init__(self, csv_path, seq_len, label_col, train_mode=False,
                 split_seed=42, split_ratio=0.8):
        df = pd.read_csv(csv_path)

        # Handle both object dtype and pandas StringDtype (ArrowDtype etc.)
        if pd.api.types.is_string_dtype(df[label_col]) or df[label_col].dtype == object:
            df[label_col] = (df[label_col] != "Benign").astype(int)

        if train_mode:
            benign = df[df[label_col] == 0]
            if len(benign) == 0:
                raise ValueError(
                    f"No benign samples found in '{csv_path}'. "
                    f"Check that the label column '{label_col}' contains 'Benign' entries."
                )
            # 80/20 split on benign samples (deterministic via seed)
            rng = np.random.RandomState(split_seed)
            indices = rng.permutation(len(benign))
            n_train = int(len(benign) * split_ratio)
            train_idx = indices[:n_train]
            df = benign.iloc[train_idx].reset_index(drop=True)
        else:
            # For test: use ALL data (benign + attacks),
            # but exclude training benign to avoid data leakage
            benign_mask = df[label_col] == 0
            benign_df = df[benign_mask]
            attack_df = df[~benign_mask]
            # Reproduce same split to identify held-out benign
            rng = np.random.RandomState(split_seed)
            indices = rng.permutation(len(benign_df))
            n_train = int(len(benign_df) * split_ratio)
            test_idx = indices[n_train:]
            test_benign = benign_df.iloc[test_idx].reset_index(drop=True)
            df = pd.concat([test_benign, attack_df], ignore_index=True)

        self.labels = df[label_col].values
        df = df.drop(columns=[label_col])

        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Drop constant/useless features
        drop_cols = [c for c in DROP_FEATURES if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)

        # ── Derived rate features ──
        # Bytes and packets per second are strong DDoS indicators.
        dur = df["Flow Duration"].values.copy() if "Flow Duration" in df.columns else None
        if dur is not None:
            dur_s = dur / 1e6  # microseconds → seconds
            dur_s = np.where(dur_s < 1e-6, 1e-6, dur_s)  # avoid div-by-zero
            if "Total Fwd Packets" in df.columns:
                df["Fwd Packets/s"] = df["Total Fwd Packets"].values / dur_s
            if "Total Backward Packets" in df.columns:
                df["Bwd Packets/s"] = df["Total Backward Packets"].values / dur_s
            if "Fwd Packets Length Total" in df.columns:
                df["Fwd Bytes/s"] = df["Fwd Packets Length Total"].values / dur_s
            if "Bwd Packets Length Total" in df.columns:
                df["Bwd Bytes/s"] = df["Bwd Packets Length Total"].values / dur_s

        self.feature_names = df.columns.tolist()

        data = df.values.astype(np.float32)
        # Clean any inf/nan from derived features
        data = np.where(np.isfinite(data), data, 0.0)

        # Log-transform heavy-tailed features before normalization.
        # sign(x) * log1p(|x|) preserves sign for negative values.
        # Includes derived rate features which are also heavy-tailed.
        log_set = set(LOG_FEATURES) | {"Fwd Packets/s", "Bwd Packets/s", "Fwd Bytes/s", "Bwd Bytes/s"}
        log_mask = np.array([c in log_set for c in self.feature_names])
        self._log_mask = log_mask
        if log_mask.any():
            data[:, log_mask] = np.sign(data[:, log_mask]) * np.log1p(np.abs(data[:, log_mask]))

        if train_mode:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0) + 1e-8
            np.savez("checkpoints/norm.npz", mean=self.mean, std=self.std,
                     log_mask=log_mask)
        else:
            stats = np.load("checkpoints/norm.npz")
            self.mean, self.std = stats["mean"], stats["std"]

        self.data = (data - self.mean) / self.std
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def to_device(self, device):
        """Pre-load entire dataset to GPU to eliminate CPU→GPU transfer."""
        self._device = device
        self._data_tensor = torch.tensor(self.data, device=device)
        self._label_tensor = torch.tensor(self.labels, device=device)
        return self

    def __getitem__(self, idx):
        if hasattr(self, '_data_tensor'):
            x = self._data_tensor[idx:idx + self.seq_len]
            y = self._label_tensor[idx + self.seq_len - 1]
            return x, y
        x = self.data[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]
        return torch.tensor(x), torch.tensor(y)

def get_loader(csv, seq_len, label, batch, shuffle, train_mode, device=None):
    ds = FlowDataset(csv, seq_len, label, train_mode)
    if device is not None and device.type == 'cuda':
        ds.to_device(device)
        # Data already on GPU — no pin_memory or workers needed
        return DataLoader(ds, batch_size=batch, shuffle=shuffle), ds
    return DataLoader(ds, batch_size=batch, shuffle=shuffle), ds
