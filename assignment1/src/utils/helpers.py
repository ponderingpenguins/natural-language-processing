"""
Helper functions for reporting model performance.
"""

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.decomposition import TruncatedSVD  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore
from umap import UMAP  # type: ignore

from .config import TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def report_stats(
    cfg: TrainingConfig, model_name: str, y_true: Any, y_pred: Any
) -> None:
    """
    Report accuracy, macro-F1, and confusion matrix. Also saves the classification report to a text file.
    Args:
    - model_name: Name of the model (for logging and file naming).
    - y_true: True labels.
    - y_pred: Predicted labels.
    Returns:
    - None (logs the results and saves the classification report to a file).
    """
    logger.info("Results for %s:", model_name)
    logger.info(classification_report(y_true, y_pred))

    # save classification report to file
    report_path = Path(cfg.output_dir) / f"{model_name}_classification_report.txt"
    with report_path.open("w") as f:
        report = classification_report(y_true, y_pred, output_dict=False)
        f.write(report)
    logger.info("Saved classification report to %s", report_path)
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_true, y_pred))


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    label_names: dict[int, str],
    title: str,
    output_path: str | Path,
) -> None:
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names.values(),
        yticklabels=label_names.values(),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.savefig(output_path)
    plt.close(fig)


def plot_tfidf_clusters(
    X,
    y,
    label_names: dict[int, str],
    output_dir: str | Path,
    n_samples: int = 5000,
    seed: int = 42,
    file_name_prefix: str = "tfidf",
) -> None:
    """Visualise TF-IDF vectors in 2-D with PCA, t-SNE, and UMAP."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Subsample
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(n_samples, X.shape[0]), replace=False)
    x_sub, y_sub = X[idx], np.array(y)[idx]

    # Pre-reduce with SVD once - shared by all projections
    n_svd = min(50, x_sub.shape[1])
    x_reduced = TruncatedSVD(n_components=n_svd, random_state=seed).fit_transform(x_sub)

    palette = sns.color_palette("tab10", n_colors=len(label_names))

    def _save_plot(coords, title, xlabel, ylabel, suffix):
        fig, ax = plt.subplots(figsize=(9, 7))
        for label_id, label_name in label_names.items():
            mask = y_sub == label_id
            ax.scatter(
                *coords[mask].T,
                s=6,
                alpha=0.4,
                label=label_name,
                color=palette[label_id - 1],
            )
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(markerscale=4)
        fig.tight_layout()
        path = output_dir / f"{file_name_prefix}_{suffix}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Saved %s", path)

    projections = [
        (
            "pca",
            lambda: TruncatedSVD(n_components=2, random_state=seed).fit_transform(
                x_sub
            ),
            "PCA projection of TF-IDF vectors",
            "PC1",
            "PC2",
        ),
        (
            "tsne",
            lambda: TSNE(
                n_components=2,
                perplexity=30,
                learning_rate="auto",
                init="pca",
                random_state=seed,
            ).fit_transform(x_reduced),
            "t-SNE projection of TF-IDF vectors",
            "$t$-SNE dim 1",
            "$t$-SNE dim 2",
        ),
        (
            "umap",
            lambda: UMAP(n_components=2, random_state=seed).fit_transform(x_reduced),
            "UMAP projection of TF-IDF vectors",
            "UMAP dim 1",
            "UMAP dim 2",
        ),
    ]

    for suffix, project, title, xlabel, ylabel in projections:
        logger.info("Computing %s projection...", suffix.upper())
        _save_plot(project(), title, xlabel, ylabel, suffix)


def plot_prediction_map(
    x_data,
    y_true,
    y_pred,
    label_names: dict[int, str],
    title: str,
    output_path: str | Path,
    seed: int = 42,
    n_samples: int = 5000,
) -> None:
    """
    Scatter plot in 2-D (t-SNE): correct predictions are small & faded,
    misclassifications are large with a black edge so they pop out.

    Uses the same SVD -> t-SNE pipeline as plot_tfidf_clusters so the
    two plots are directly comparable.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    # Sub-sample
    rng = np.random.default_rng(seed)
    n = min(n_samples, x_data.shape[0])
    idx = rng.choice(x_data.shape[0], size=n, replace=False)
    x_sub = x_data[idx]
    yt_sub = y_true_arr[idx]
    yp_sub = y_pred_arr[idx]

    # Reduce: SVD -> t-SNE (same pipeline as plot_tfidf_clusters)
    n_svd = min(50, x_sub.shape[1])
    x_reduced = TruncatedSVD(n_components=n_svd, random_state=seed).fit_transform(x_sub)
    coords = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    ).fit_transform(x_reduced)

    correct = yt_sub == yp_sub
    palette = sns.color_palette("tab10", n_colors=len(label_names))

    fig, ax = plt.subplots(figsize=(9, 7))

    # Correct predictions — small, faded
    for label_id, label_name in label_names.items():
        mask = correct & (yt_sub == label_id)
        ax.scatter(
            *coords[mask].T,
            s=6,
            alpha=0.3,
            color=palette[label_id - 1],
            label=label_name,
        )  # type: ignore

    # Misclassifications — large, opaque, black edge
    mis = ~correct
    ax.scatter(
        *coords[mis].T,
        s=10,
        alpha=0.9,
        c="none",
        edgecolors="black",
        linewidths=0.8,
        label="Misclassified",
    )  # type: ignore

    ax.set(title=title, xlabel="$t$-SNE dim 1", ylabel="$t$-SNE dim 2")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Prediction map saved to %s", output_path)
