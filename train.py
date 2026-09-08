import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.data.feature_extraction import FeatureExtractor
from src.data.windowing import WindowBuilder
from src.data.dataset import create_dataloaders
from src.models.world_model import build_world_model
from src.models.baseline import BaselineModel
from src.evaluation.metrics import evaluate_model, print_evaluation_report


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits, _ = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return total_loss / len(loader), correct / total


def main():
    parser = argparse.ArgumentParser(description="Train World Model for Network Attack Forecasting")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", required=True, help="Path to CIC-IDS CSV file")
    parser.add_argument("--output", default="checkpoints/")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    print("Loading and processing data...")
    extractor = FeatureExtractor()
    df, feature_cols, label_col = extractor.process_file(args.data)
    print(f"Loaded {len(df)} flows with {len(feature_cols)} features")

    print("Building time windows...")
    window_builder = WindowBuilder(
        window_size=config["data"]["window_size"],
        slide_size=config["data"]["slide_size"],
    )
    X, y, feature_names = window_builder.build_windows(df, feature_cols, label_col=label_col)
    print(f"Created {len(X)} windows of size {config['data']['window_size']}s")

    n = len(X)
    train_end = int(n * config["data"]["train_ratio"])
    val_end = int(n * (config["data"]["train_ratio"] + config["data"]["val_ratio"]))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    X_train_scaled, X_test_scaled = extractor.normalize(X_train.reshape(-1, X_train.shape[-1]))
    X_train_scaled = X_train_scaled.reshape(X_train.shape)
    _, X_val_scaled = extractor.normalize(X_train.reshape(-1, X_train.shape[-1]), X_val.reshape(-1, X_val.shape[-1]))
    X_val_scaled = X_val_scaled.reshape(X_val.shape)
    _, X_test_scaled = extractor.normalize(X_train.reshape(-1, X_train.shape[-1]), X_test.reshape(-1, X_test.shape[-1]))
    X_test_scaled = X_test_scaled.reshape(X_test.shape)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("Training baseline model (Logistic Regression)...")
    baseline = BaselineModel()
    baseline.fit(X_train_scaled, y_train)
    baseline_metrics = baseline.evaluate(X_test_scaled, y_test)
    print_evaluation_report(baseline_metrics, "Baseline (Logistic Regression)")

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test,
        batch_size=config["training"]["batch_size"],
    )

    print(f"Building {config['model']['type'].upper()} World Model...")
    # Features doubled: mean + std aggregation per window
    model_input_size = 2 * len(feature_cols)
    model = build_world_model(
        model_type=config["model"]["type"],
        input_size=model_input_size,
        num_classes=6,
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0

    print("\nStarting training...")
    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:3d}/{config['training']['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": config,
                "feature_cols": feature_cols,
            }, Path(args.output) / "best_model.pt")
            print("  -> Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["patience"]:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(Path(args.output) / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    world_model_metrics = evaluate_model(model, test_loader, device)
    print_evaluation_report(world_model_metrics, f"World Model ({config['model']['type'].upper()})")

    improvement = world_model_metrics["f1_macro"] - baseline_metrics["f1_macro"]
    print(f"\nImprovement over baseline: {improvement:+.4f} F1 (macro)")
    print(f"Baseline F1: {baseline_metrics['f1_macro']:.4f}")
    print(f"World Model F1: {world_model_metrics['f1_macro']:.4f}")

    import json
    results = {
        "baseline": baseline_metrics,
        "world_model": world_model_metrics,
        "improvement_f1": improvement,
        "model_type": config["model"]["type"],
        "dataset": args.data,
    }
    with open(Path(args.output) / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}/evaluation_results.json")
    print("Training complete!")


if __name__ == "__main__":
    main()
