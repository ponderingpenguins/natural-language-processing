from .helpers import logger


def hello() -> str:
    logger.info("Called hello function")
