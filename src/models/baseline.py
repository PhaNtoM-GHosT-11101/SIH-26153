import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class BaselineModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_timesteps, n_features = X.shape
        X_flat = X.reshape(n_samples, -1)
        X_scaled = self.scaler.fit_transform(X_flat)
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_timesteps, n_features = X.shape
        X_flat = X.reshape(n_samples, -1)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples, n_timesteps, n_features = X.shape
        X_flat = X.reshape(n_samples, -1)
        X_scaled = self.scaler.transform(X_flat)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, y_pred),
            "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
            "f1_weighted": f1_score(y, y_pred, average="weighted", zero_division=0),
            "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        }
