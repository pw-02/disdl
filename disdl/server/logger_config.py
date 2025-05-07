import logging

def configure_logger():
    # Set the log levels for specific loggers to WARNING
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("hydra.core.utils").setLevel(logging.WARNING)

    # Configure the root logger with DEBUG level and a simple format
    # logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Create and configure a logger named "SUPER"
    logger = logging.getLogger("DISDL")
    logger.setLevel(logging.DEBUG)
    return logger

logger = configure_logger()
