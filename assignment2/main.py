"""Main training script for Assignment 2"""

import sys

from omegaconf import OmegaConf
from penguinlp.config import TrainingConfig
from penguinlp.data import load_data
from penguinlp.helpers import logger
from utils.tokanizer import build_tokenizer, load_tokenizer, save_tokenizer


def fooberino(cfg: TrainingConfig) -> None:
    """
    Placeholder function for the main training logic.

    Args:
        cfg: Training configuration.
    Returns:
        None
    """
    # TODO: Implement the training logic here
    data = load_data(cfg)

    # load tokenizer if it exists, otherwise build and save a new one
    try:
        tokenizer = load_tokenizer(cfg.tokenizer_path)
        logger.info("Tokenizer loaded successfully.")
    except FileNotFoundError:
        logger.warning("Tokenizer not found. Building a new one.")

        tokenizer = build_tokenizer(
            data["train"], tokenizer_type=cfg.tokenizer_type, vocab_size=1000
        )
        logger.info("Vocabulary built with %d tokens", len(tokenizer.vocab))

        save_tokenizer(tokenizer, cfg.tokenizer_path)
    breakpoint()


def main() -> None:
    """main function"""
    logger.info("Assignment 2: Training a TF-IDF classifier")
    cfg = OmegaConf.structured(TrainingConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    try:
        cfg = TrainingConfig(**cfg)  # type: ignore
    except TypeError:  # pylint: disable=broad-exception-raised
        logger.exception("Error\n\nUsage: python main.py")
        sys.exit(1)

    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("Data loaded successfully. Starting training...")
    fooberino(cfg)  # TODO: Replace with actual training function


if __name__ == "__main__":
    main()
