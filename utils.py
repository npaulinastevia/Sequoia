from abc import ABC

import gym
import logging
from functools import wraps
from pathlib import Path
root_logger = logging.getLogger("")
from typing import Any, Callable, Dict, Iterable, List, TypeVar, Union

def get_logger(name: str, level: int = None) -> logging.Logger:
    """Gets a logger for the given file. Sets a nice default format.
    TODO: figure out if we should add handlers, etc.
    """
    name_is_path: bool = False
    try:
        p = Path(name)
        if p.exists():
            name = str(p.absolute().relative_to(Path.cwd()).as_posix())
            name_is_path = True
    except:
        pass
    from sys import argv

    logger = root_logger.getChild(name)

    debug_flags: List[str] = ["-d", "--debug", "-vv", "-vvv" "--verbose"]

    if level is None and any(v in argv for v in debug_flags):
        level = logging.DEBUG
    if level is None:
        level = logging.INFO
    logger.setLevel(level)

    # if the name is already something like foo.py:256
    # if not name_is_path and name[-1].isdigit():
    #     formatter = logging.Formatter('%(asctime)s, %(levelname)-8s log [%(name)s] %(message)s')
    # sh = logging.StreamHandler(sys.stdout)
    # sh.setFormatter(formatter)
    # sh.setLevel(level)
    # logger.addHandler(sh)
    # logger = logging.getLogger(name)
    # tqdm_handler = TqdmLoggingHandler()
    # tqdm_handler.setLevel(level)
    # logger.addHandler(tqdm_handler)
    return logger
class MayCloseEarly(gym.Wrapper, ABC):
    """ABC for Wrappers that may close an environment early depending on some
    conditions.

    WIP: Also prevents calling `step` and `reset` on a closed env.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._is_closed: bool = False

    def is_closed(self) -> bool:
        # First, make sure that we're not 'overriding' the 'is_closed' of the
        # wrapped environment.
        if hasattr(self.env, "is_closed"):
            assert callable(self.env.is_closed)
            self._is_closed = self.env.is_closed()
        return self._is_closed

    def closed_error_message(self) -> str:
        """Return the error message to use when attempting to use the closed env.

        This can be useful for wrappers that close when a given condition is reached,
        e.g. a number of episodes has been performed, which could return a more relevant
        message here.
        """
        return "Env is closed"

    def reset(self, **kwargs):
        if self.is_closed():
            raise gym.error.ClosedEnvironmentError(
                f"Can't call `reset()`: {self.closed_error_message()}"
            )
        return super().reset(**kwargs)

    def step(self, action):
        if self.is_closed():
            raise gym.error.ClosedEnvironmentError(
                f"Can't call `step()`: {self.closed_error_message()}"
            )
        return super().step(action)

    def close(self) -> None:
        if self.is_closed():
            # TODO: Prevent closing an environment twice?
            return
            # raise gym.error.ClosedEnvironmentError(self.closed_error_message())
        self.env.close()
        self._is_closed = True
from gym import Space, spaces

