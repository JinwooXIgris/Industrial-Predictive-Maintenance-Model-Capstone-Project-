"""
model_evaluation.py
Module for evaluation metrics calculation, confusion matrices, ROC/PR curves,
threshold optimization (maximizing F1-Score), learning curves, and feature
importance visualization.
"""

import os
import sys
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance

# Import project configurations
try:
    import config
except ModuleNotFoundError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import config

# Set plot style for publication quality
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "legend.fontsize": 10
})


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Computes comprehensive classification metrics.
    """
    # Raw confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_true, y_prob)
    
    # Advanced diagnostic metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Equal to Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall/Sensitivity": recall,
        "F1-Score": f1,
        "AUC-ROC": auc_roc,
        "Specificity": specificity,
        "False Positive Rate": fpr,
        "False Negative Rate": fnr
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Tuple[float, float]:
    """
    Finds the probability threshold that maximizes F1-Score on validation set.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_f1 = 0.0
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return float(best_thresh), float(best_f1)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: str = None
) -> None:
    """
    Generates a publication-quality confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Format text annotations with counts and percentages
    labels = np.asarray([
        f"{count}\n({pct:.1f}%)"
        for count, pct in zip(cm.flatten(), cm_percent.flatten())
    ]).reshape(2, 2)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Failure", "Failure"],
        yticklabels=["No Failure", "Failure"],
        ax=ax,
        annot_kws={"size": 12, "weight": "bold"}
    )
    ax.set_xlabel("Predicted Label", labelpad=10)
    ax.set_ylabel("True Label", labelpad=10)
    ax.set_title(f"Confusion Matrix: {model_name}", pad=15)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    optimal_threshold: float = None,
    save_path_prefix: str = None
) -> None:
    """
    Plots Receiver Operating Characteristic (ROC) and Precision-Recall (PR) curves.
    """
    # 1. ROC Curve
    fpr_vals, tpr_vals, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(fpr_vals, tpr_vals, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    ax1.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("Receiver Operating Characteristic (ROC)")
    ax1.legend(loc="lower right")
    
    # 2. Precision-Recall Curve
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    
    ax2.plot(rec, prec, color="forestgreen", lw=2, label="Precision-Recall Curve")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall (PR) Curve")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    
    # Annotate optimal threshold if provided
    if optimal_threshold is not None:
        # Find index of threshold closest to optimal
        idx = np.argmin(np.abs(thresholds - optimal_threshold))
        if idx < len(rec) and idx < len(prec):
            ax2.plot(rec[idx], prec[idx], 'ro', markersize=8, label=f"Opt Thresh = {optimal_threshold:.2f}")
            ax2.annotate(
                f"T={optimal_threshold:.2f}",
                xy=(rec[idx], prec[idx]),
                xytext=(rec[idx]-0.15, prec[idx]-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6)
            )
            
    ax2.legend(loc="lower left")
    
    plt.suptitle(f"Performance Curves: {model_name}", y=1.02)
    plt.tight_layout()
    
    if save_path_prefix:
        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
        plt.savefig(f"{save_path_prefix}_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def extract_feature_importance(
    model: Any,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: List[str]
) -> pd.Series:
    """
    Extracts feature importances depending on the model architecture.
    Uses Gini importance for Tree/Forest, coefs for LR, and permutation
    importance as a robust fallback for boosting models.
    """
    model_class = model.__class__.__name__
    
    if hasattr(model, "feature_importances_"):
        # Tree-based Gini importance
        importances = model.feature_importances_
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
    elif hasattr(model, "coef_"):
        # Logistic Regression coefficients (magnitude indicates importance)
        importances = np.abs(model.coef_[0])
        # Normalize to sum to 1 to compare easily
        importances = importances / np.sum(importances)
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)
        
    else:
        # Fallback: Permutation Importance
        print(f"[INFO] Calculating permutation importance for {model_class} (takes a few seconds)...")
        result = permutation_importance(
            model, X_val, y_val,
            n_repeats=5,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        importances = result.importances_mean
        # Handle negative importance bounds gracefully
        importances = np.maximum(importances, 0)
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def plot_feature_importance(
    importances: pd.Series,
    model_name: str,
    top_n: int = 8,
    save_path: str = None
) -> None:
    """
    Generates a horizontal bar chart of feature importances.
    """
    top_features = importances.head(top_n)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=top_features.values,
        y=top_features.index,
        palette="viridis",
        hue=top_features.index,
        legend=False,
        ax=ax
    )
    ax.set_xlabel("Relative Importance Score")
    ax.set_ylabel("Features")
    ax.set_title(f"Top {top_n} Feature Importance: {model_name}", pad=15)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_learning_curves(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    save_path: str = None
) -> None:
    """
    Generates learning curves to detect overfitting and assess sample efficiency.
    """
    print(f"[INFO] Computing learning curves for {model_name}...")
    train_sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE),
        scoring="f1",
        n_jobs=-1,
        random_state=config.RANDOM_STATE
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sizes, train_mean, 'o-', color="tomato", label="Training F1-Score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="tomato")
    
    ax.plot(train_sizes, val_mean, 's-', color="steelblue", label="Cross-Validation F1-Score")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="steelblue")
    
    ax.set_xlabel("Training Set Size (samples)")
    ax.set_ylabel("F1-Score")
    ax.set_title(f"Learning Curves: {model_name}", pad=15)
    ax.legend(loc="best")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
