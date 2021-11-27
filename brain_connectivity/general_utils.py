import logging

formatter = logging.Formatter(
    fmt="%(asctime)s %(name)-15s %(levelname)-8s %(message)s",
    datefmt="%d-%m-%Y %H:%M",
)

# Keeps track of already taken names, so that we don't attach multiple handlers to a single logger.
logger_names = set()
loggers = []

# Console log.
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger = logging.getLogger("bc")
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)


def get_logger(name: str, filename: str):
    "Returns logger that logs both to file and console at once."
    child_logger = logger.getChild(name)
    if name in logger_names:
        child_logger.warning(
            f"Attempting to get logger with already taken name (name: {name}, filename: {filename}). "
            + "Not attaching new file handler."
        )
    else:
        file_handler = logging.FileHandler(filename=filename, mode="a")
        file_handler.setFormatter(formatter)
        # Log everything into file.
        file_handler.setLevel(logging.DEBUG)
        child_logger.addHandler(file_handler)
    logger_names.add(name)
    loggers.append(child_logger)

    return child_logger


def end_logging():
    global logger_names, loggers
    logger_names = set()
    for l in loggers:
        for h in l.handlers:
            h.close()
    loggers = []
