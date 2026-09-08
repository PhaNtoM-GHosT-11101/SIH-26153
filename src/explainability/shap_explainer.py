import numpy as np
import shap
import torch
from typing import Optional, Callable
from src.data.dataset import STAGE_NAMES


class SHAPExplainer:
    def __init__(self, model: torch.nn.Module, feature_names: list[str], device: str = "cpu"):
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.model.eval()
        self._explainer = None

    def _model_predict_flat(self, X_flat: np.ndarray) -> np.ndarray:
        # X_flat: (n_samples, sequence_length * n_feat)
        seq_len = self._seq_len
        n_feat = self._n_feat
        X = X_flat.reshape(-1, seq_len, n_feat)
        x = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits, _ = self.model(x)
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def explain_sequence(self, sequence: np.ndarray, num_samples: int = 50) -> dict:
        # sequence: (sequence_length, features) where features includes mean+std
        self._seq_len = sequence.shape[0]
        n_feat = sequence.shape[1]
        self._n_feat = n_feat
        seq_flat = sequence.reshape(-1).astype(np.float32)

        background = np.random.randn(num_samples, seq_flat.shape[0]).astype(np.float32) * 0.1
        background += seq_flat

        explainer = shap.KernelExplainer(self._model_predict_flat, background)
        shap_values = explainer.shap_values(seq_flat, nsamples=num_samples)

        if isinstance(shap_values, list):
            per_class = [np.abs(sv) for sv in shap_values]
            mean_abs = np.mean(per_class, axis=0)
        else:
            mean_abs = np.abs(shap_values)

        # Aggregate over timesteps AND collapse mean/std halves back to base features
        base_feat = len(self.feature_names)  # number of named base features
        feature_importance = []
        for f in range(base_feat):
            # indices in flat vector for this base feature across timesteps, mean & std halves
            idxs = []
            for t in range(self._seq_len):
                idxs.append(t * n_feat + f)          # mean half
                idxs.append(t * n_feat + base_feat + f)  # std half
            imp = float(mean_abs[idxs].mean())
            feature_importance.append({"feature": self.feature_names[f], "importance": imp})
        feature_importance.sort(key=lambda x: x["importance"], reverse=True)

        probs = self._model_predict_flat(seq_flat[None, :])[0]
        predicted_stage = STAGE_NAMES[np.argmax(probs)]

        return {
            "predicted_stage": predicted_stage,
            "probabilities": {STAGE_NAMES[i]: float(probs[i]) for i in range(len(STAGE_NAMES))},
            "top_driving_features": feature_importance[:10],
        }

    def explain_trajectory(self, sequences: np.ndarray, step: int = 0) -> list[dict]:
        return [self.explain_sequence(s) for s in sequences]
