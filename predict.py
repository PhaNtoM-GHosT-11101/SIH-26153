import argparse
import yaml
import torch
import numpy as np
import json
from pathlib import Path

from src.data.feature_extraction import FeatureExtractor
from src.data.windowing import WindowBuilder
from src.models.world_model import build_world_model
from src.prediction.engine import PredictionEngine
from src.explainability.shap_explainer import SHAPExplainer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Run predictions with trained World Model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--data", required=True, help="Path to CSV file for inference")
    parser.add_argument("--k-steps", type=int, default=5, help="Forecast horizon")
    parser.add_argument("--output", default="predictions/")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    extractor = FeatureExtractor()
    df, feature_cols, label_col = extractor.process_file(args.data)

    print("Building windows...")
    window_builder = WindowBuilder(
        window_size=config["data"]["window_size"],
        slide_size=config["data"]["slide_size"],
    )
    X, y, feature_names = window_builder.build_windows(df, feature_cols, label_col=label_col)

    X_scaled, _ = extractor.normalize(X.reshape(-1, X.shape[-1]))
    X_scaled = X_scaled.reshape(X.shape)

    print("Loading model...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_input_size = 2 * len(feature_cols)
    model = build_world_model(
        model_type=config["model"]["type"],
        input_size=model_input_size,
        num_classes=6,
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    engine = PredictionEngine(model, device, sequence_length=10)

    print(f"Generating predictions for {len(X_scaled)} windows...")
    timeline = engine.generate_timeline(X_scaled, k_steps=args.k_steps)

    print("Running explainability analysis...")
    explainer = SHAPExplainer(model, feature_names, device)
    sequences, _ = engine._build_sequences(X_scaled)

    sample_indices = np.linspace(0, len(sequences) - 1, min(5, len(sequences)), dtype=int)
    explanations = []
    for idx in sample_indices:
        exp = explainer.explain_sequence(sequences[idx])
        exp["window_index"] = int(idx)
        explanations.append(exp)

    results = {
        "total_windows": len(timeline),
        "k_steps": args.k_steps,
        "stage_distribution": {},
        "timeline_summary": [],
        "explanations": [],
    }

    stage_counts = {}
    for entry in timeline:
        stage = entry["predicted_stage"]
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    results["stage_distribution"] = stage_counts

    for entry in timeline[:50]:
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

    for exp in explanations:
        clean_exp = {
            "window_index": exp["window_index"],
            "predicted_stage": exp["predicted_stage"],
            "probabilities": exp["probabilities"],
            "top_driving_features": exp["top_driving_features"][:5],
        }
        results["explanations"].append(clean_exp)

    output_path = Path(args.output) / "prediction_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nPredictions saved to {output_path}")
    print(f"Stage distribution: {stage_counts}")

    print("\n--- Sample Predictions ---")
    for entry in timeline[:5]:
        print(f"  Window {entry['window_index']}: {entry['predicted_stage']} "
              f"(conf: {entry['confidence']:.3f})")
        for f in entry["forecast"]:
            print(f"    +{f['step']}s: {f['predicted_stage']} (conf: {f['confidence']:.3f})")


if __name__ == "__main__":
    main()
