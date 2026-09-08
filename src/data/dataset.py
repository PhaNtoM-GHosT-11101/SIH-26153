import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


STAGE_NAMES = [
    "Benign",
    "Reconnaissance",
    "Initial Access",
    "Lateral Movement",
    "Command and Control",
    "Exfiltration",
]


def aggregate_window(window: np.ndarray) -> np.ndarray:
    """Aggregate a 2D window (n_flows, n_features) into a 1D feature vector.
    Concatenates mean and std per feature for richer state representation."""
    if window.ndim == 1:
        return window.astype(np.float32)
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float32)


class NetworkDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 10):
        self.seq_len = sequence_length
        # Aggregate each window to a feature vector
        agg = np.array([aggregate_window(w) for w in X], dtype=np.float32)
        self.X = torch.tensor(agg, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.X) - self.seq_len + 1)

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_label = self.y[idx + self.seq_len - 1]
        return x_seq, y_label


class TrajectoryDataset(Dataset):
    def __init__(self, X: np.ndarray, y_trajectory: np.ndarray, sequence_length: int = 10):
        self.seq_len = sequence_length
        agg = np.array([aggregate_window(w) for w in X], dtype=np.float32)
        self.X = torch.tensor(agg, dtype=torch.float32)
        self.y = torch.tensor(y_trajectory, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.X) - self.seq_len + 1)

    def __getitem__(self, idx):
        x_seq = self.X[idx: idx + self.seq_len]
        y_traj = self.y[idx + self.seq_len - 1]
        return x_seq, y_traj


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 64,
    sequence_length: int = 10,
    trajectory: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if trajectory:
        train_ds = TrajectoryDataset(X_train, y_train, sequence_length)
        val_ds = TrajectoryDataset(X_val, y_val, sequence_length)
        test_ds = TrajectoryDataset(X_test, y_test, sequence_length)
    else:
        train_ds = NetworkDataset(X_train, y_train, sequence_length)
        val_ds = NetworkDataset(X_val, y_val, sequence_length)
        test_ds = NetworkDataset(X_test, y_test, sequence_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
