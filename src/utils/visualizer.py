"""src/utils/visualizer.py - Plotting and visualization utilities."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def save_fig(fig, path: str, dpi: int = 150):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str = None, title: str = "Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_feature_importance(feature_names, importances, save_path: str = None, top_n: int = 20):
    indices = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color="steelblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_clusters_2d(X, labels, method: str = "pca", save_path: str = None, title: str = "Cluster Visualization"):
    """Reduce to 2D and plot cluster assignments."""
    print(f"[Info] Reducing to 2D using {method.upper()}...")
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    X_2d = reducer.fit_transform(X)
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        label_name = str(label) if label != -1 else "Noise"
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color], label=label_name,
                   alpha=0.6, s=20, edgecolors="none")
    
    ax.set_title(f"{title} ({method.upper()})", fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_training_history(history, save_path: str = None):
    """Plot CNN training history (loss and accuracy)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history.get("val_loss", []), label="Val Loss")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    
    ax2.plot(history.history.get("accuracy", []), label="Train Acc")
    ax2.plot(history.history.get("val_accuracy", []), label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    
    plt.suptitle("CNN Training History", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_class_distribution(y, class_names=None, save_path: str = None, title: str = "Class Distribution"):
    unique, counts = np.unique(y, return_counts=True)
    if class_names:
        labels = [class_names[i] if i < len(class_names) else str(i) for i in unique]
    else:
        labels = [str(i) for i in unique]
    
    fig, ax = plt.subplots(figsize=(max(8, len(unique)), 5))
    bars = ax.bar(labels, counts, color=sns.color_palette("husl", len(unique)))
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_elbow_curve(k_values, inertias, save_path: str = None):
    """Plot elbow curve for K-Means cluster selection."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, inertias, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Curve for K-Means", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig
