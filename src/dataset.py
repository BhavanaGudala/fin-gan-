import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FlowDataset(Dataset):
    def __init__(self, csv_path, seq_len, label_col, train_mode=False):
        df = pd.read_csv(csv_path)

        # Handle both object dtype and pandas StringDtype (ArrowDtype etc.)
        if pd.api.types.is_string_dtype(df[label_col]) or df[label_col].dtype == object:
            df[label_col] = (df[label_col] != "Benign").astype(int)

        if train_mode:
            df = df[df[label_col] == 0]   # Benign only
            if len(df) == 0:
                raise ValueError(
                    f"No benign samples found in '{csv_path}'. "
                    f"Check that the label column '{label_col}' contains 'Benign' entries."
                )

        self.labels = df[label_col].values
        df = df.drop(columns=[label_col])

        df = df.select_dtypes(include=[np.number])
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        data = df.values.astype(np.float32) #conversion dataframes to arrays

        if train_mode:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0) + 1e-8
            np.savez("checkpoints/norm.npz", mean=self.mean, std=self.std)
        else:
            stats = np.load("checkpoints/norm.npz")
            self.mean, self.std = stats["mean"], stats["std"]

        self.data = (data - self.mean) / self.std
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]
        return torch.tensor(x), torch.tensor(y)

def get_loader(csv, seq_len, label, batch, shuffle, train_mode):
    ds = FlowDataset(csv, seq_len, label, train_mode)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle), ds
