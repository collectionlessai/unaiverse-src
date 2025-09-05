"""
       █████  █████ ██████   █████           █████ █████   █████ ██████████ ███████████    █████████  ██████████
      ░░███  ░░███ ░░██████ ░░███           ░░███ ░░███   ░░███ ░░███░░░░░█░░███░░░░░███  ███░░░░░███░░███░░░░░█
       ░███   ░███  ░███░███ ░███   ██████   ░███  ░███    ░███  ░███  █ ░  ░███    ░███ ░███    ░░░  ░███  █ ░ 
       ░███   ░███  ░███░░███░███  ░░░░░███  ░███  ░███    ░███  ░██████    ░██████████  ░░█████████  ░██████   
       ░███   ░███  ░███ ░░██████   ███████  ░███  ░░███   ███   ░███░░█    ░███░░░░░███  ░░░░░░░░███ ░███░░█   
       ░███   ░███  ░███  ░░█████  ███░░███  ░███   ░░░█████░    ░███ ░   █ ░███    ░███  ███    ░███ ░███ ░   █
       ░░████████   █████  ░░█████░░████████ █████    ░░███      ██████████ █████   █████░░█████████  ██████████
        ░░░░░░░░   ░░░░░    ░░░░░  ░░░░░░░░ ░░░░░      ░░░      ░░░░░░░░░░ ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░░░░░░ 
                 A Collectionless AI Project (https://collectionless.ai)
                 Registration/Login: https://unaiverse.io
                 Code Repositories:  https://github.com/collectionlessai/
                 Main Developers:    Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler

LOG_FOLDER = "unaiverse/networking/p2p/logs"

def setup_logger(module_name: str, when: str = 'midnight', backup_count: int = 7) -> logging.Logger:
    """
    Sets up a logger for a specific module with a timed rotating file handler.
    Args:
        module_name (str): The name of the module for which the logger is being set up.
        when (str, optional): Specifies the type of interval for log rotation.
                              Defaults to 'midnight'. Common values include 'S', 'M', 'H', 'D', 'midnight', etc.
        backup_count (int, optional): The number of backup log files to keep. Defaults to 7.
    Returns:
        logging.Logger: A configured logger instance for the specified module.
    Notes:
        - Log files are stored in a predefined folder (`LOG_FOLDER`).
        - Log rotation occurs based on the specified interval (`when`).
        - The logger writes logs in UTF-8 encoding and uses UTC time by default.
        - The log format includes timestamp, log level, filename, line number, and the log message.
        - If the logger for the module already exists, it will not add duplicate handlers.
    """
    do_log = False  # Stefano

    # Ensure log folder exists
    if do_log:
        os.makedirs(LOG_FOLDER, exist_ok=True)

    # Log file base path (without date suffix)
    log_base_filename = os.path.join(LOG_FOLDER, f"{module_name}.log")

    # Create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    if do_log and not logger.handlers:

        # Create rotating file handler
        handler = TimedRotatingFileHandler(
            log_base_filename,
            when=when,
            interval=1,
            backupCount=backup_count,
            encoding='utf-8',
            utc=True  # Set to False if you want local time
        )

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Avoid logging to root handlers
        logger.propagate = False

    if not do_log:
        logger.handlers.clear()
        logger.addHandler(logging.NullHandler())

    return logger
