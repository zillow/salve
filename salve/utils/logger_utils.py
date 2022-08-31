"""Utilities for logging.
"""
import logging
import os
import sys
from logging import Logger

from salve.utils.datetime_utils import generate_datetime_string


def get_logger() -> Logger:
    """Getter for the main logger."""
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger


def setup_file_logger(home_dir: str, program_name: str):
    """ """
    date_str = generate_datetime_string()
    log_output_fpath = f"{home_dir}/logging/{program_name}_{date_str}.log"

    os.makedirs(f"{home_dir}/logging", exist_ok=True)

    logging.basicConfig(
        format="[%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=log_output_fpath,
        level=logging.INFO,
    )
    logging.debug("Init Debug")
    logging.info("Init Info")
    logging.warning("Init Warning")
    logging.critical("Init Critical")
