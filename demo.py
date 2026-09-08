#!/usr/bin/env python3
"""
Demo script for SIH-26153 — generates sample predictions for video recording.
Run this after training to create demo output.
"""
import numpy as np
import json
from pathlib import Path

from src.data.dataset import STAGE_NAMES


def generate_synthetic_demo():
    """Generate synthetic demo data showing the system's capabilities."""
    np.random.seed(42)

    n_windows = 50
    timeline = []

    # Simulate an attack progression
    stages_sequence = (
        ["Benign"] * 10 +
        ["Reconnaissance"] * 8 +
        ["Initial Access"] * 6 +
        ["Lateral Movement"] * 8 +
        ["Command and Control"] * 10 +
        ["Exfiltration"] * 5 +
        ["Benign"] * 3
    )

    for i in range(n_windows):
        true_stage = stages_sequence[i] if i < len(stages_sequence) else "Benign"

        probs = np.random.dirichlet(np.ones(6) * 0.5)
        stage_idx = STAGE_NAMES.index(true_stage) if true_stage in STAGE_NAMES else 0
        probs[stage_idx] = np.random.uniform(0.7, 0.95)
        probs = probs / probs.sum()

        forecast = []
        for k in range(1, 6):
            future_idx = min(i + k, len(stages_sequence) - 1)
            future_stage = stages_sequence[future_idx]
            future_probs = np.random.dirichlet(np.ones(6) * 0.3)
            fi = STAGE_NAMES.index(future_stage) if future_stage in STAGE_NAMES else 0
            future_probs[fi] = np.random.uniform(0.6, 0.9)
            future_probs = future_probs / future_probs.sum()
            forecast.append({
                "step": k,
                "predicted_stage": future_stage,
                "confidence": float(future_probs[fi]),
            })

        entry = {
            "window_index": i,
            "predicted_stage": true_stage,
            "stage_index": stage_idx,
            "confidence": float(probs[stage_idx]),
            "probabilities": {STAGE_NAMES[j]: float(probs[j]) for j in range(6)},
            "forecast": forecast,
        }
        timeline.append(entry)

    return timeline


def generate_sample_explanations():
    """Generate sample SHAP explanations."""
    feature_groups = {
        "Reconnaissance": [
            {"feature": "SYN Flag Count", "importance": 0.85},
            {"feature": "Dst Port", "importance": 0.72},
            {"feature": "Flow Packets/s", "importance": 0.68},
            {"feature": "Flow Duration", "importance": 0.55},
            {"feature": "Protocol", "importance": 0.42},
        ],
        "Initial Access": [
            {"feature": "Flow Bytes/s", "importance": 0.78},
            {"feature": "Fwd Packet Length Mean", "importance": 0.65},
            {"feature": "Init_Win_bytes_forward", "importance": 0.58},
            {"feature": "Bwd Packet Length Max", "importance": 0.45},
            {"feature": "ACK Flag Count", "importance": 0.38},
        ],
        "Lateral Movement": [
            {"feature": "Dst Port", "importance": 0.90},
            {"feature": "Flow Duration", "importance": 0.75},
            {"feature": "Fwd PSH Flags", "importance": 0.62},
            {"feature": "Flow IAT Mean", "importance": 0.55},
            {"feature": "Total Fwd Packets", "importance": 0.48},
        ],
        "Command and Control": [
            {"feature": "Flow Packets/s", "importance": 0.82},
            {"feature": "Flow IAT Std", "importance": 0.70},
            {"feature": "Bwd Packet Length Mean", "importance": 0.60},
            {"feature": "Down/Up Ratio", "importance": 0.52},
            {"feature": "Fwd Header Length", "importance": 0.40},
        ],
        "Exfiltration": [
            {"feature": "Total Length of Bwd Packets", "importance": 0.88},
            {"feature": "Flow Bytes/s", "importance": 0.75},
            {"feature": "Bwd Packet Length Max", "importance": 0.65},
            {"feature": "Init_Win_bytes_backward", "importance": 0.50},
            {"feature": "Packet Length Variance", "importance": 0.42},
        ],
    }
    return feature_groups


def main():
    output_dir = Path("predictions/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic demo timeline...")
    timeline = generate_synthetic_demo()

    print("Generating sample explanations...")
    explanations = generate_sample_explanations()

    results = {
        "demo": True,
        "total_windows": len(timeline),
        "k_steps": 5,
        "stage_distribution": {},
        "timeline_summary": [],
        "explanations": [],
    }

    stage_counts = {}
    for entry in timeline:
        stage = entry["predicted_stage"]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    results["stage_distribution"] = stage_counts

    for entry in timeline:
        summary = {
            "window": entry["window_index"],
            "stage": entry["predicted_stage"],
            "confidence": round(entry["confidence"], 4),
            "forecast": [
                {"step": f["step"], "stage": f["predicted_stage"], "confidence": round(f["confidence"], 4)}
                for f in entry["forecast"]
            ],
        }
        results["timeline_summary"].append(summary)

    for stage, features in explanations.items():
        results["explanations"].append({
            "stage": stage,
            "top_driving_features": features,
        })

    output_path = output_dir / "prediction_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Demo data saved to {output_path}")
    print(f"\nStage distribution:")
    for stage, count in sorted(stage_counts.items()):
        print(f"  {stage}: {count} windows")

    print(f"\nTo view the demo:")
    print(f"  streamlit run src/demo/app.py")


if __name__ == "__main__":
    main()
