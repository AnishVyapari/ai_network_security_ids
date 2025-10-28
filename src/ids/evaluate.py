"""
Evaluation utilities: metrics and plots for IDS models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        pos_scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, pos_scores)
        except Exception:
            metrics['roc_auc'] = float('nan')
    return metrics


def plot_confusion_matrix(y_true, y_pred, out_path: Path | str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_curve(y_true, y_proba, out_path: Path | str):
    from sklearn.metrics import RocCurveDisplay
    pos_scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
    disp = RocCurveDisplay.from_predictions(y_true, pos_scores)
    plt.title('ROC Curve')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
