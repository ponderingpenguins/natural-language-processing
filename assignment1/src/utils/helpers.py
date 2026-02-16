import logging

from sklearn.metrics import classification_report, confusion_matrix

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
    logger.info(f"Results for {model_name}:")
    logger.info(classification_report(y_true, y_pred))
    logger.info("Confusion Matrix:")
    logger.info(confusion_matrix(y_true, y_pred))
    logger.info(confusion_matrix(y_true, y_pred))
