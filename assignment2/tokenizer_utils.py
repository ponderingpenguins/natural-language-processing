"""Tokenizer setup helpers."""

from penguinlp.config import TrainingConfig
from penguinlp.helpers import logger
from utils.tokanizer import build_tokenizer, load_tokenizer, save_tokenizer


def setup_tokenizer(cfg: TrainingConfig, train_data) -> object:
    """Setup and return tokenizer, either by loading or building new one.

    Args:
        cfg: Training configuration.
        train_data: Training dataset for building tokenizer if needed.

    Returns:
        Tokenizer object.
    """
    try:
        tokenizer = load_tokenizer(cfg.tokenizer_path)
        logger.info("Tokenizer loaded successfully.")
    except FileNotFoundError:
        logger.warning("Tokenizer not found. Building a new one.")
        tokenizer = build_tokenizer(
            train_data, tokenizer_type=cfg.tokenizer_type, vocab_size=1000
        )
        logger.info("Vocabulary built with %d tokens", len(tokenizer.vocab))
        save_tokenizer(tokenizer, cfg.tokenizer_path)

    return tokenizer
