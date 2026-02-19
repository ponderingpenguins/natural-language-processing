"""
Helper functions for reporting model performance.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.decomposition import TruncatedSVD  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


def report_stats(model_name, y_true, y_pred):
    """Report accuracy, macro-F1, and confusion matrix."""
    logger.info("Results for %s:", model_name)
    logger.info(classification_report(y_true, y_pred))
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_true, y_pred))


def plot_confusion_matrix(
    y_true,
    y_pred,
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
) -> None:
    """Visualise TF-IDF vectors in 2-D with PCA and t-SNE.

    A random subsample of n_samples documents is used so that
    t-SNE runs in reasonable time.

    Saves two PNG files to output_dir:
        - tfidf_pca.png   (PCA projection)
        - tfidf_tsne.png  (t-SNE projection)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n_total = X.shape[0]
    if n_total > n_samples:
        idx = rng.choice(n_total, size=n_samples, replace=False)
    else:
        idx = np.arange(n_total)

    x_sub = X[idx]
    y_sub = np.array(y)[idx]

    palette = sns.color_palette("tab10", n_colors=len(label_names))

    # PCA (via TruncatedSVD, works on sparse matrices)
    logger.info("Computing PCA (TruncatedSVD) projection …")
    svd = TruncatedSVD(n_components=2, random_state=seed)
    x_pca = svd.fit_transform(x_sub)

    fig, ax = plt.subplots(figsize=(9, 7))
    for label_id, label_name in label_names.items():
        mask = y_sub == label_id
        ax.scatter(
            x_pca[mask, 0],
            x_pca[mask, 1],
            s=6,
            alpha=0.4,
            label=label_name,
            color=palette[label_id - 1],
        )
    ax.set_title("TF-IDF vectors — PCA (TruncatedSVD) projection")
    ax.set_xlabel(f"PC 1 ({svd.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC 2 ({svd.explained_variance_ratio_[1]:.1%} var)")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(output_dir / "tfidf_pca.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_dir / "tfidf_pca.png")

    # t-SNE (reduce to 50-D with SVD first for speed)
    logger.info("Computing t-SNE projection (this may take a minute) …")
    n_components_svd = min(50, x_sub.shape[1])
    svd50 = TruncatedSVD(n_components=n_components_svd, random_state=seed)
    x_reduced = svd50.fit_transform(x_sub)

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    x_tsne = tsne.fit_transform(x_reduced)

    fig, ax = plt.subplots(figsize=(9, 7))
    for label_id, label_name in label_names.items():
        mask = y_sub == label_id
        ax.scatter(
            x_tsne[mask, 0],
            x_tsne[mask, 1],
            s=6,
            alpha=0.4,
            label=label_name,
            color=palette[label_id - 1],
        )
    ax.set_title("TF-IDF vectors — t-SNE projection")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(markerscale=4)
    fig.tight_layout()
    fig.savefig(output_dir / "tfidf_tsne.png", dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_dir / "tfidf_tsne.png")
