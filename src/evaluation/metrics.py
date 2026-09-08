import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix
)
from src.data.dataset import STAGE_NAMES


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def evaluate_model(model, dataloader, device: str = "cpu") -> dict:
    import torch
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits, _ = model(batch_x)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = compute_all_metrics(y_true, y_pred)
    metrics["classification_report"] = classification_report(
        y_true, y_pred,
        target_names=STAGE_NAMES[:len(np.unique(np.concatenate([y_true, y_pred])))],
        zero_division=0,
        output_dict=True,
    )
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def print_evaluation_report(metrics: dict, model_name: str = "Model"):
    print(f"\n{'='*60}")
    print(f"  Evaluation Report: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  F1 (Macro):         {metrics['f1_macro']:.4f}")
    print(f"  F1 (Weighted):      {metrics['f1_weighted']:.4f}")
    print(f"  Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"  Recall (Macro):     {metrics['recall_macro']:.4f}")
    print(f"{'='*60}\n")
