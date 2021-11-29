import logging

formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)-7s %(name)-15s %(message)s",
    datefmt="%d-%m-%Y %H:%M",
)

# Keeps track of already taken names, so that we don't attach multiple handlers to a single logger.
loggers = {}

# Root logger.
logger = logging.getLogger("bc")
logger.setLevel(logging.DEBUG)

# Console log.
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
# Don't show debug messages.
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)


def get_logger(name: str, filename: str):
    "Returns logger that logs both to file and console at once."
    child_logger = logger.getChild(name)
    if name in loggers:
        child_logger.warning(
            f"Attempting to get logger with already taken name (name: {name}, filename: {filename}). "
            + "Not attaching new file handler."
        )
    else:
        file_handler = logging.FileHandler(filename=filename, mode="a")
        file_handler.setFormatter(formatter)
        child_logger.addHandler(file_handler)
        loggers[name] = child_logger

    return child_logger


def close_logger(name: str):
    logger = loggers.pop(name)
    handlers = logger.handlers
    for handler in handlers:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def close_all_loggers():
    global loggers
    logger_names = list(loggers.keys())
    for name in logger_names:
        close_logger(name)
