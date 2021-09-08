import logging
from pathlib import Path
from typing import List, Optional


def set_root_logger(log_level: int, log_file: Optional[Path]) -> None:
    logging_handlers: List[logging.Handler] = []
    if log_file is None:
        logging_handlers.append(logging.StreamHandler())
    else:
        logging_handlers.append(logging.FileHandler(filename=log_file))

    logging.captureWarnings(True)

    logging.basicConfig(
        format="[%(levelname)-8s][%(asctime)s][%(process) 5d][%(name)s]:"
        ' %(message)s -- File "%(pathname)s", line %(lineno)d,'
        " in %(funcName)s,",
        level=log_level,
        handlers=logging_handlers,
    )
