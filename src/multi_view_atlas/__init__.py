import logging
from importlib.metadata import version

from rich.logging import RichHandler

from . import pl, pp, tl, utils

logger = logging.getLogger(__name__)
logger.propagate = False
ch = RichHandler(level=logging.INFO, show_path=False, show_time=False)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.removeHandler(logger.handlers[0])


__version__ = version("multi-view-atlas")
