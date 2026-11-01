"""
Visualization Module
======================
Publication-quality plots for the hackathon demo.
Includes: Loss curves, confusion matrices, ROC curves,
          feature importance, quantum circuit diagrams,
          kernel matrix heatmaps, and comparison charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

# ── Theming ──────────────────────────────────────────────────────────────────
QUANTUM_PALETTE = {
    "primary":    "#6C63FF",   # Quantum purple
    "secondary":  "#00D4AA",   # Teal
    "classical":  "#FF6B6B",   # Red-coral
    "quantum":    "#6C63FF",   # Purple
    "accent":     "#FFD166",   # Yellow
    "bg":         "#0F0F23",   # Dark background
    "surface":    "#1A1A35",   # Card background
    "text":       "#E8E8FF",   # Light text
    "grid":       "#2A2A50",   # Grid lines
}

plt.rcParams.update({
    "figure.facecolor": QUANTUM_PALETTE["bg"],
    "axes.facecolor":   QUANTUM_PALETTE["surface"],
    "axes.edgecolor":   QUANTUM_PALETTE["grid"],
    "axes.labelcolor":  QUANTUM_PALETTE["text"],
    "xtick.color":      QUANTUM_PALETTE["text"],
    "ytick.color":      QUANTUM_PALETTE["text"],
    "text.color":       QUANTUM_PALETTE["text"],
    "grid.color":       QUANTUM_PALETTE["grid"],
    "grid.alpha":       0.4,
    "font.family":      "DejaVu Sans",
    "axes.titlesize":   14,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "figure.titlesize": 16,
})


def _savefig(fig, path: str, dpi: int = 150):
    """Save figure with tight layout."""
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=QUANTUM_PALETTE["bg"])
    plt.close(fig)
    print(f"  [OK] Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────

def plot_training_loss(loss_history: list, val_acc_history: list = None,
                       title: str = "VQC Training Curve",
                       save_path: str = None):
    """Plot training loss and optional validation accuracy over epochs."""
    fig, axes = plt.subplots(1, 2 if val_acc_history else 1,
                              figsize=(14 if val_acc_history else 7, 5))
    if val_acc_history is None:
        axes = [axes]

    # Loss
    ax = axes[0]
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, color=QUANTUM_PALETTE["quantum"],
            linewidth=2, label="Training Loss")
    # Smooth trend
    if len(loss_history) > 10:
        window = max(3, len(loss_history) // 10)
        smoothed = pd.Series(loss_history).rolling(window, center=True).mean()
        ax.plot(epochs, smoothed, color=QUANTUM_PALETTE["accent"],
                linewidth=2, linestyle="--", label="Smoothed")
    ax.set_title("Training Loss (Binary Cross-Entropy)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    # Val Accuracy
    if val_acc_history:
        ax2 = axes[1]
        val_epochs = [10 * (i + 1) for i in range(len(val_acc_history))]
        ax2.plot(val_epochs, val_acc_history,
                 color=QUANTUM_PALETTE["secondary"],
                 linewidth=2, marker="o", markersize=6, label="Val Accuracy")
        ax2.axhline(y=max(val_acc_history), color=QUANTUM_PALETTE["accent"],
                    linestyle=":", alpha=0.7,
                    label=f"Best: {max(val_acc_history):.4f}")
        ax2.set_title("Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)
        ax2.legend()
        ax2.grid(True)

    fig.suptitle(title, fontweight="bold", y=1.02)

    if save_path:
        _savefig(fig, save_path)
    else:
        plt.show()


def plot_confusion_matrix(cm: np.ndarray, labels: list,
                           title: str = "Confusion Matrix",
                           save_path: str = None):
    """Plot confusion matrix with percentage annotations."""
    fig, ax = plt.subplots(figsize=(7, 6))

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    im = ax.imshow(cm_norm, cmap="RdPu", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i, j]
            pct = 100 * cm_norm[i, j]
            color = "white" if cm_norm[i, j] > 0.5 else QUANTUM_PALETTE["text"]
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=12,
                    fontweight="bold", color=color)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    if save_path:
        _savefig(fig, save_path)
    else:
        plt.show()


def plot_roc_curves(models_results: dict, save_path: str = None):
    """
    Plot ROC curves for multiple models on the same axes.

    Args:
        models_results: dict mapping model_name → (y_true, y_score)
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = [QUANTUM_PALETTE["quantum"], QUANTUM_PALETTE["secondary"],
              QUANTUM_PALETTE["classical"], QUANTUM_PALETTE["accent"],
              "#FF9F43", "#5F27CD", "#00D2D3"]

    for idx, (name, (y_true, y_score)) in enumerate(models_results.items()):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        color = colors[idx % len(colors)]
        lw = 3 if "Quantum" in name or "VQC" in name or "QSVM" in name else 1.5
        ls = "-" if "Quantum" in name or "VQC" in name or "QSVM" in name else "--"
        ax.plot(fpr, tpr, color=color, lw=lw, linestyle=ls,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", lw=1.5, label="Random (AUC = 0.500)")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Quantum vs Classical", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    if save_path:
        _savefig(fig, save_path)
    else:
        plt.show()


def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    """
    Grouped bar chart comparing accuracy, F1, AUC across all models.
    Quantum models highlighted with purple bars.
    """
    metrics = ["accuracy_mean", "f1_mean", "auc_mean"]
    metric_labels = ["Accuracy", "F1 Score", "AUC-ROC"]

    models = list(results_df.index)
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 6))

    colors_by_metric = [
        QUANTUM_PALETTE["quantum"],
        QUANTUM_PALETTE["secondary"],
        QUANTUM_PALETTE["accent"],
    ]

    for m_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if metric not in results_df.columns:
            continue
        values = results_df[metric].values
        std_col = metric.replace("_mean", "_std")
        errors = results_df[std_col].values if std_col in results_df.columns else None

        bars = ax.bar(x + m_idx * width, values, width,
                      label=label, color=colors_by_metric[m_idx],
                      alpha=0.85, edgecolor="white", linewidth=0.5)

        if errors is not None:
            ax.errorbar(x + m_idx * width, values, yerr=errors,
                        fmt="none", color="white", capsize=4, linewidth=1.5)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8,
                    color=QUANTUM_PALETTE["text"])

    # Highlight quantum models
    for i, model in enumerate(models):
        if any(kw in model for kw in ["Quantum", "VQC", "QSVM", "Hybrid"]):
            ax.axvspan(x[i] - 0.15, x[i] + 0.9, alpha=0.08,
                       color=QUANTUM_PALETTE["quantum"])

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison: Quantum vs Classical Baselines",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Add legend for quantum highlight
    q_patch = mpatches.Patch(color=QUANTUM_PALETTE["quantum"],
                              alpha=0.2, label="Quantum Model")
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles + [q_patch], labels_leg + ["Quantum Model"],
              fontsize=10, loc="upper right")

    if save_path:
        _savefig(fig, save_path)
    else:
        plt.show()


def plot_feature_importance(importance_df: pd.DataFrame, save_path: str = None):
    """Horizontal bar chart of feature importances."""
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = [QUANTUM_PALETTE["quantum"] if i < 4 else QUANTUM_PALETTE["classical"]
              for i in range(len(importance_df))]

    bars = ax.barh(importance_df["feature"], importance_df["importance"],
                   color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)

    for bar, val in zip(bars, importance_df["importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    ax.set_xlabel("Feature Importance (RF Gini)", fontsize=12)
    ax.set_title("Feature Importance Analysis\n(Purple = used as quantum qubits)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    q_patch = mpatches.Patch(color=QUANTUM_PALETTE["quantum"], label="→ Quantum Circuit Features")
    c_patch = mpatches.Patch(color=QUANTUM_PALETTE["classical"], label="Classical-only Features")
    ax.legend(handles=[q_patch, c_patch], fontsize=10)

    ax.invert_yaxis()

    if save_path:
        _savefig(fig, save_path)
    else:
        plt.show()


def plot_kernel_matrix(K: np.ndarray, labels: np.ndarray = None,
                       title: str = "Quantum Kernel Matrix",
                       save_path: str = None):
    """Heatmap of the quantum kernel matrix, sorted by class label."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw kernel matrix
    im1 = axes[0].imshow(K, cmap="plasma", aspect="auto",
                          interpolation="nearest")
    plt.colorbar(im1, ax=axes[0], fraction=0.046)
    axes[0].set_title(f"{title}\n(Raw order)", fontsize=12)
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Sample index")

    # Sorted by label
    if labels is not None:
        sort_idx = np.argsort(labels)
        K_sorted = K[sort_idx][:, sort_idx]
        im2 = axes[1].imshow(K_sorted, cmap="plasma", aspect="auto",
                              interpolation="nearest")
        plt.colorbar(im2, ax=axes[1], fraction=0.046)
        axes[1].set_title(f"{title}\n(Sorted by class label)", fontsize=12)
        axes[1].set_xlabel("Sample index (sorted)")

        # Class separator line
        n_class0 = int(np.sum(labels == 0))
        for ax in axes:
            ax.axvline(n_class0 - 0.5, color="white", lw=2, ls="--", alpha=0.8)
            ax.axhline(n_class0 - 0.5, color="white", lw=2, ls="--", alpha=0.8)
    else:
        axes[1].axis("off")

    fig.suptitle(title, fontsize=15, fontweight="bold")

    if save_path:
        _savefig(fig, save_path)
    else:
        plt.show()


def plot_data_overview(df: pd.DataFrame, save_path: str = None):
    """Overview of OASIS dataset: distributions and correlations."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("OASIS Dataset Overview — Alzheimer's Disease Classification",
                 fontsize=16, fontweight="bold", y=1.01)

    # CDR distribution
    ax1 = fig.add_subplot(2, 4, 1)
    cdr_counts = df["CDR"].value_counts().sort_index()
    bars = ax1.bar(cdr_counts.index.astype(str), cdr_counts.values,
                   color=[QUANTUM_PALETTE["secondary"], QUANTUM_PALETTE["accent"],
                           QUANTUM_PALETTE["classical"], QUANTUM_PALETTE["quantum"]][:len(cdr_counts)],
                   edgecolor="white", linewidth=0.5)
    ax1.set_title("CDR Distribution")
    ax1.set_xlabel("CDR Score")
    ax1.set_ylabel("Count")
    for bar, val in zip(bars, cdr_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(val), ha="center", fontsize=11, fontweight="bold")

    # Age distribution by class
    ax2 = fig.add_subplot(2, 4, 2)
    dementia = df[df["CDR"] > 0]["AGE"]
    healthy = df[df["CDR"] == 0]["AGE"]
    ax2.hist(healthy, bins=12, alpha=0.7, label="Healthy (CDR=0)",
             color=QUANTUM_PALETTE["secondary"], edgecolor="white")
    ax2.hist(dementia, bins=12, alpha=0.7, label="Dementia (CDR>0)",
             color=QUANTUM_PALETTE["classical"], edgecolor="white")
    ax2.set_title("Age Distribution by Class")
    ax2.set_xlabel("Age")
    ax2.legend(fontsize=8)

    # MMSE vs CDR
    ax3 = fig.add_subplot(2, 4, 3)
    df_scatter = df.dropna(subset=["MMSE", "nWBV", "CDR"])
    scatter = ax3.scatter(df_scatter["MMSE"], df_scatter["nWBV"],
                          c=df_scatter["CDR"], cmap="plasma",
                          alpha=0.8, s=60, edgecolors="white", linewidth=0.5)
    plt.colorbar(scatter, ax=ax3, label="CDR")
    ax3.set_title("MMSE vs nWBV")
    ax3.set_xlabel("MMSE Score")
    ax3.set_ylabel("Normalized WBV")

    # Feature correlation heatmap
    ax4 = fig.add_subplot(2, 4, 4)
    feature_cols = ["AGE", "EDUC", "SES", "MMSE", "eTIV", "ASF", "nWBV", "CDR"]
    corr_data = df[feature_cols].dropna().corr()
    mask = np.zeros_like(corr_data)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr_data, ax=ax4, cmap="RdBu_r", center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                mask=None)
    ax4.set_title("Feature Correlations")
    ax4.tick_params(axis="x", rotation=45, labelsize=8)
    ax4.tick_params(axis="y", rotation=0, labelsize=8)

    # nWBV distribution
    ax5 = fig.add_subplot(2, 4, 5)
    for cdr_val, label, color in [
        (0,   "Healthy",  QUANTUM_PALETTE["secondary"]),
        (0.5, "MCI",      QUANTUM_PALETTE["accent"]),
        (1.0, "Mild AD",  QUANTUM_PALETTE["classical"]),
    ]:
        subset = df[df["CDR"] == cdr_val]["nWBV"].dropna()
        if len(subset) > 1:
            ax5.hist(subset, bins=8, alpha=0.7, label=label, color=color, edgecolor="white")
    ax5.set_title("Brain Volume (nWBV) by CDR")
    ax5.set_xlabel("Normalized WBV")
    ax5.legend(fontsize=8)

    # Sex distribution
    ax6 = fig.add_subplot(2, 4, 6)
    sex_cdr = df.groupby(["SEX", "CDR"]).size().unstack(fill_value=0)
    sex_cdr.plot(kind="bar", ax=ax6, color=[
        QUANTUM_PALETTE["secondary"], QUANTUM_PALETTE["accent"],
        QUANTUM_PALETTE["classical"]][:len(sex_cdr.columns)],
        edgecolor="white", linewidth=0.5)
    ax6.set_title("Sex vs CDR")
    ax6.set_xlabel("Sex")
    ax6.tick_params(axis="x", rotation=0)
    ax6.legend(title="CDR", fontsize=8)

    # ASF vs eTIV
    ax7 = fig.add_subplot(2, 4, 7)
    df_asf = df.dropna(subset=["eTIV", "ASF", "CDR"])
    scatter2 = ax7.scatter(df_asf["eTIV"], df_asf["ASF"],
                           c=(df_asf["CDR"] > 0).astype(int),
                           cmap="coolwarm", alpha=0.8, s=60,
                           edgecolors="white", linewidth=0.5)
    plt.colorbar(scatter2, ax=ax7, label="Dementia")
    ax7.set_title("eTIV vs ASF")
    ax7.set_xlabel("eTIV (mm³)")
    ax7.set_ylabel("ASF")

    # Education vs MMSE
    ax8 = fig.add_subplot(2, 4, 8)
    for cdr_val, label, color in [(0, "Healthy", QUANTUM_PALETTE["secondary"]),
                                   (0.5, "MCI", QUANTUM_PALETTE["accent"]),
                                   (1.0, "Dementia", QUANTUM_PALETTE["classical"])]:
        subset = df[df["CDR"] == cdr_val].dropna(subset=["EDUC", "MMSE"])
        if len(subset) > 0:
            ax8.scatter(subset["EDUC"], subset["MMSE"],
                        label=label, color=color, alpha=0.7, s=50,
                        edgecolors="white", linewidth=0.5)
    ax8.set_title("Education vs MMSE")
    ax8.set_xlabel("Education Level")
    ax8.set_ylabel("MMSE Score")
    ax8.legend(fontsize=8)

    if save_path:
        _savefig(fig, save_path)
    else:
        plt.show()


def plot_quantum_advantage_radar(results_df: pd.DataFrame, save_path: str = None):
    """
    Radar (spider) chart comparing quantum and classical models
    across multiple dimensions: accuracy, F1, AUC, interpretability, speed.
    """
    categories = ["Accuracy", "F1 Score", "AUC-ROC"]
    n_cats = len(categories)
    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8),
                            subplot_kw=dict(polar=True))
    ax.set_facecolor(QUANTUM_PALETTE["surface"])
    fig.patch.set_facecolor(QUANTUM_PALETTE["bg"])

    colors = {
        "Hybrid VQC":      QUANTUM_PALETTE["quantum"],
        "QSVM":            QUANTUM_PALETTE["secondary"],
    }
    classical_color = QUANTUM_PALETTE["classical"]

    for model_name in results_df.index:
        row = results_df.loc[model_name]
        values = [
            row.get("accuracy_mean", 0),
            row.get("f1_mean", 0),
            row.get("auc_mean", 0),
        ]
        values += values[:1]

        color = colors.get(model_name, classical_color)
        lw = 3 if model_name in colors else 1.5
        ls = "-" if model_name in colors else "--"
        ax.plot(angles, values, color=color, linewidth=lw, linestyle=ls, label=model_name)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=13, color=QUANTUM_PALETTE["text"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"],
                        color=QUANTUM_PALETTE["text"], size=9)
    ax.grid(color=QUANTUM_PALETTE["grid"], alpha=0.5)
    ax.set_title("Quantum vs Classical — Performance Radar",
                 pad=20, fontsize=14, fontweight="bold",
                 color=QUANTUM_PALETTE["text"])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15),
              facecolor=QUANTUM_PALETTE["surface"],
              edgecolor=QUANTUM_PALETTE["grid"],
              labelcolor=QUANTUM_PALETTE["text"])

    if save_path:
        _savefig(fig, save_path)
    else:
        plt.show()
