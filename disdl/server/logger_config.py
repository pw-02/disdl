import logging

def configure_logger(name: str = "DISDL", log_file: str = "disd.log") -> logging.Logger:
    # Silence noisy libraries
    for noisy in ["PIL", "botocore", "urllib3", "boto3", "hydra.core.utils"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # Only configure handlers if not already set (avoids duplicate logs)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            # fmt='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
            fmt='[%(name)s][%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)

        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        # logger.addHandler(stream_handler)

    return logger

def configure_simulation_logger() -> logging.Logger:
    return configure_logger(name="DISDL-SIM", log_file="simulation.log")



# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='disdlapp.log',         # Log file name
#     filemode='w'                # Overwrite the file each run; use 'a' to append
# )