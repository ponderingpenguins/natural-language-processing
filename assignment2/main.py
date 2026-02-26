"""Main training script for Assignment 2"""

import sys

from omegaconf import OmegaConf
from penguinlp.config import TrainingConfig
from penguinlp.helpers import logger


def fooberino(cfg: TrainingConfig) -> None:
    """
    Placeholder function for the main training logic.

    Args:
        cfg: Training configuration.
    Returns:
        None
    """
    # TODO: Implement the training logic here
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
    fooberino(cfg)  # TODO: Replace with actual training function


if __name__ == "__main__":
    main()
