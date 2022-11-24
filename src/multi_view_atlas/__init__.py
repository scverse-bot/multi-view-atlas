import logging
from importlib.metadata import version

from rich.logging import RichHandler

from . import pl, pp, tl, utils
from .config import set_options

# logger = logging.getLogger("multi_view_atlas")
# logger.propagate = False
# logger.setLevel("INFO")
# ch = RichHandler(level=logging.INFO, show_path=False, show_time=False)
# formatter = logging.Formatter("%(message)s")
# ch.setFormatter(formatter)
# logger.addHandler(ch)


__version__ = version("multi_view_atlas")
