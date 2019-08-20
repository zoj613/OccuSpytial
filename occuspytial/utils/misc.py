import logging
import logging.config
import os
from pathlib import Path

from typing import Any, List, Optional, Sequence, Tuple
import warnings

import numpy as np  # type: ignore

try:
    import toml
except ImportError:
    warnings.showwarning(
        "The toml package is not installed. Logging configuration through "
        "the toml config file can not be used until it is installed.",
        category=ImportWarning,
        filename=__name__,
        lineno=27
    )

logger = logging.getLogger(__name__)


class CustomDict(dict):
    """
    A custom dictionary that supports indexing via its keys. The index
    can be *args or any iterable e.g., numpy array, list, tuple.
    """
    # a mix-in class for group-indexing W and y dictionaries
    def slice(self, *keys: Sequence[Any]) -> np.ndarray:
        """Take in a sequence of keys as input and return the corres-
        ponding values as one concatenated numpy array over axis 0.

        Args:
            *keys (Sequence[Any]): variable length sequence of keys.

        Returns:
            np.ndarray: a stacked numpy array of values that correspond
            to the key arguments. If the values of numpy arrays then
            they are stacked row-wise.
        """
        try:
            out = [self[k] for k in keys]
        except TypeError:  # if input is not hashable
            out = [self[k] for k in keys[0]]
        return np.concatenate(tuple(out))


# adapated from:
# https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
def log_config(
    default_path: str = 'log_config.toml',
    default_level: int = logging.DEBUG,
    env_key: str = 'LOG_CFG'
):
    """Setup logging configuration"""
    path = Path(default_path)
    value = os.getenv(env_key, None)
    if value:
        path = Path(value)
    log_directory = Path('logs')
    if not log_directory.exists():
        log_directory.mkdir()
    if path.exists():
        try:
            _config = toml.load(path)
            logging.config.dictConfig(_config)
        except NameError:
            logging.basicConfig(level=default_level)
            logging.error(
                "toml module could not be imported, thus custom logging config"
                " failed. Basic logging config will be used instead."
            )
    else:
        logging.basicConfig(level=default_level)
        logging.error(
            "The specified log config file path does not exist. "
            "Basic logging config will be used instead."
        )
