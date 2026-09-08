import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


ATTACK_STAGE_MAP = {
    "Benign": "Benign",
    "Bot": "Command and Control",
    "DDoS": "Initial Access",
    "DoS Golden Eye": "Initial Access",
    "DoS Hulk": "Initial Access",
    "DoS Slowloris": "Initial Access",
    "FTP-Patator": "Initial Access",
    "SSH-Patator": "Initial Access",
    "Web Attack - Brute Force": "Initial Access",
    "Web Attack - SQL Injection": "Initial Access",
    "Web Attack - XSS": "Initial Access",
    "Infiltration": "Lateral Movement",
    "Heartbleed": "Command and Control",
    "PortScan": "Reconnaissance",
    "Network Scan": "Reconnaissance",
    "Portmap": "Reconnaissance",
    "Backdoor": "Exfiltration",
    "Keylogger": "Exfiltration",
    "Ransomware": "Exfiltration",
    "Trojan": "Command and Control",
}


class FeatureExtractor:
    def __init__(self, feature_cols: Optional[list[str]] = None):
        self.feature_cols = feature_cols
        self.scaler = None
        self.label_col = None

    def load_csv(self, path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path, low_memory=False)
        df = df.dropna(axis=1, how="all")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        return df

    def identify_columns(self, df: pd.DataFrame) -> tuple[list[str], str]:
        exclude = {"Label", "label", "Timestamp", "Flow ID", "Source IP",
                    "Destination IP", "Source Port", "Destination Port"}
        if self.feature_cols:
            cols = [c for c in self.feature_cols if c in df.columns]
        else:
            cols = [c for c in df.columns if c not in exclude and df[c].dtype in ("float64", "int64", "float32", "int32")]

        label_col = None
        for candidate in ("Label", "label", " Label"):
            if candidate in df.columns:
                label_col = candidate
                break

        return cols, label_col

    def map_labels_to_stages(self, df: pd.DataFrame, label_col: str) -> pd.Series:
        # Direct stage names (already normalized) pass through unchanged
        direct = set(ATTACK_STAGE_MAP.values()) | {"Benign"}
        mapped = df[label_col].map(ATTACK_STAGE_MAP)
        # For labels that are already stage names, keep them
        already_stage = df[label_col].isin(direct)
        mapped[already_stage] = df[label_col][already_stage]
        return mapped.fillna("Benign")

    def extract_timestamps(self, df: pd.DataFrame) -> Optional[pd.Series]:
        for col in ("Timestamp", "timestamp"):
            if col in df.columns:
                return pd.to_datetime(df[col], errors="coerce")
        return None

    def normalize(self, X_train: np.ndarray, X_test: Optional[np.ndarray] = None):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test) if X_test is not None else None
        return X_train_scaled, X_test_scaled

    def process_file(self, path: str | Path) -> tuple[pd.DataFrame, list[str], str]:
        df = self.load_csv(path)
        feature_cols, label_col = self.identify_columns(df)
        if label_col:
            df["attack_stage"] = self.map_labels_to_stages(df, label_col)
        return df, feature_cols, label_col
