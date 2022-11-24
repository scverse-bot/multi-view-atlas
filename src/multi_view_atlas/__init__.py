import logging
from importlib.metadata import version

from rich.logging import RichHandler

from . import pl, pp, tl, utils

logger = logging.getLogger("multi_view_atlas")
logger.propagate = False
ch = RichHandler(level=logging.INFO, show_path=False, show_time=False)
formatter = logging.Formatter("%(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


__version__ = version("multi-view-atlas")
