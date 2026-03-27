import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
)

for noisy in (
    "urllib3",
    "requests",
    "filelock",
    "huggingface_hub",
    "datasets",
    "transformers",
    "httpx",
):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
