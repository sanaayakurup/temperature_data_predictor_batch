import logging 
import logging
import sys

def setup_logger(name, log_file, level=logging.DEBUG):
    """
    Sets up a logger that logs to a file and also captures uncaught exceptions globally.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already set up
    if not logger.handlers:
        # Convert string to logging level if needed
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.DEBUG)

        logger.setLevel(level)

        # Create file handler
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Optional: also log to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Register global exception hook to log uncaught exceptions
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Don't log Ctrl+C interruptions
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

        sys.excepthook = handle_exception

    return logger


#def setup_logger(name,log_file,level=logging.DEBUG):
    """
    function to set up a logger to use in other scripts 
    """
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger