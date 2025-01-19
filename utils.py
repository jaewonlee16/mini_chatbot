import logging
from contextlib import contextmanager

@contextmanager
def debug_logging(level):
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield 
    finally:
        logger.setLevel(old_level)


# Add a step to print the context
def print_context(input_data):
    logging.info(input_data["context"])
    return input_data  # Ensure the data is passed to the next step in the chain