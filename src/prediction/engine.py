import torch
import numpy as np
from typing import Optional
from src.models.world_model import build_world_model
from src.data.dataset import STAGE_NAMES, aggregate_window


class PredictionEngine:
    def __init__(self, model: torch.nn.Module, device: str = "cpu", sequence_length: int = 10):
        self.model = model
        self.device = device
        self.sequence_length = sequence_length
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_type: str,
                        input_size: int, num_classes: int = 6, device: str = "cpu",
                        sequence_length: int = 10):
        model = build_world_model(model_type, input_size, num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model, device, sequence_length)

    def _build_sequences(self, windows: np.ndarray) -> np.ndarray:
        """Aggregate 2D windows to feature vectors and chunk into sequences."""
        # windows: (n_windows, n_flows, n_features) or (n_windows, n_features)
        agg = np.array([aggregate_window(w) for w in windows], dtype=np.float32)
        n = len(agg)
        if n < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} windows, got {n}")
        # Build sliding sequences of sequence_length
        seqs = []
        for i in range(self.sequence_length - 1, n):
            seqs.append(agg[i - self.sequence_length + 1: i + 1])
        return np.array(seqs, dtype=np.float32), agg

    def predict_current_state(self, sequence: np.ndarray) -> dict:
        # sequence: (sequence_length, features)
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, attn_weights = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()

        return {
            "predicted_stage": STAGE_NAMES[pred_class],
            "stage_index": pred_class,
            "probabilities": {
                STAGE_NAMES[i]: probs[0, i].item()
                for i in range(len(STAGE_NAMES))
            },
            "attention_weights": attn_weights[0].cpu().numpy().tolist(),
            "confidence": probs[0, pred_class].item(),
        }

    def forecast_k_steps(self, sequence: np.ndarray, k_steps: int = 5) -> list[dict]:
        predictions = []
        current_seq = sequence.copy()

        for step in range(k_steps):
            result = self.predict_current_state(current_seq)
            result["step"] = step + 1
            predictions.append(result)
            # Roll forward: shift sequence, append a pseudo-feature based on prediction
            new_vec = current_seq[-1].copy()
            new_vec[0] = result["stage_index"]
            current_seq = np.concatenate([current_seq[1:], new_vec[None, :]], axis=0)

        return predictions

    def generate_timeline(self, windows: np.ndarray, k_steps: int = 5) -> list[dict]:
        sequences, _ = self._build_sequences(windows)
        timeline = []
        for i, seq in enumerate(sequences):
            current = self.predict_current_state(seq)
            current["window_index"] = i + self.sequence_length - 1
            forecast = self.forecast_k_steps(seq, k_steps)
            current["forecast"] = forecast
            timeline.append(current)
        return timeline
