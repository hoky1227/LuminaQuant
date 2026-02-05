import logging
import os
from logging.handlers import RotatingFileHandler
from lumina_quant.config import BaseConfig


def setup_logging(name="lumina_quant"):
    """
    Sets up a logger with a StreamHandler and FileHandler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(BaseConfig.LOG_LEVEL)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (Rotating: 10MB limit, 5 backups)
    if not os.path.exists("logs"):
        os.makedirs("logs")

    fh = RotatingFileHandler(
        f"logs/{name}.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
