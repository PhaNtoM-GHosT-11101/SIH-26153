import numpy as np
import pandas as pd
from typing import Optional


STAGE_TO_INDEX = {
    "Benign": 0,
    "Reconnaissance": 1,
    "Initial Access": 2,
    "Lateral Movement": 3,
    "Command and Control": 4,
    "Exfiltration": 5,
}


class WindowBuilder:
    def __init__(self, window_size: int = 60, slide_size: int = 30):
        self.window_size = window_size
        self.slide_size = slide_size

    def build_windows(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        timestamps: Optional[pd.Series] = None,
        label_col: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if timestamps is not None:
            df = df.copy()
            df["_window_ts"] = timestamps
            df = df.sort_values("_window_ts")
            time_values = timestamps.sort_values().values.astype(np.int64) // 10**9
        else:
            df = df.reset_index(drop=True)
            time_values = np.arange(len(df))

        n = len(df)
        windows_X = []
        windows_y = []

        start = 0
        while start + self.window_size <= n:
            end = start + self.window_size
            window_data = df.iloc[start:end]

            features = window_data[feature_cols].values.astype(np.float32)

            if label_col and "attack_stage" in window_data.columns:
                stages = window_data["attack_stage"].values
                stage_counts = pd.Series(stages).value_counts()
                majority_stage = stage_counts.index[0]
                stage_label = STAGE_TO_INDEX.get(majority_stage, 0)
            else:
                stage_label = 0

            windows_X.append(features)
            windows_y.append(stage_label)

            start += self.slide_size

        X = np.array(windows_X, dtype=np.float32)
        y = np.array(windows_y, dtype=np.int64)

        return X, y, feature_cols

    def build_windows_with_stage_trajectory(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        k_steps: int = 5,
        timestamps: Optional[pd.Series] = None,
        label_col: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        X, y_current, _ = self.build_windows(df, feature_cols, timestamps, label_col)

        n_windows = len(X)
        y_trajectory = []

        for i in range(n_windows):
            trajectory = []
            for k in range(k_steps):
                idx = min(i + k, n_windows - 1)
                trajectory.append(y_current[idx])
            y_trajectory.append(trajectory)

        y_trajectory = np.array(y_trajectory, dtype=np.int64)
        return X, y_trajectory, feature_cols
