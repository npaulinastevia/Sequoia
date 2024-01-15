import bisect
import dataclasses
from functools import singledispatch
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
#import pytest
from gym import spaces
from gym.envs.classic_control import CartPoleEnv
from gym.vector import SyncVectorEnv
from gym.wrappers import TimeLimit

#from sequoia.common.gym_wrappers import MultiTaskEnvironment
#from sequoia.conftest import atari_py_required, monsterkong_required, param_requires_monsterkong
#from sequoia.utils.utils import dict_union

from multitaskEnv import MultiTaskEnvironment
import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import CartPoleEnv
from torch import Tensor

#from sequoia.common.spaces.named_tuple import NamedTupleSpace
#from sequoia.utils.logging_utils import get_logger


"""Test that the 'info' dict contains the task dict."""
original: CartPoleEnv = gym.make("CartPole-v0")
starting_length = original.length
starting_gravity = original.gravity

task_schedule = {
    10: dict(length=0.1),
    20: dict(length=0.2, gravity=-12.0),
    30: dict(gravity=0.9),
}
env = MultiTaskEnvironment(
    original,
    task_schedule=task_schedule,
    add_task_id_to_obs=True,
)
env.seed(123)
env.reset()

assert env.observation_space == spaces.Dict(
    x=original.observation_space,
    task_labels=spaces.Discrete(4),
)

for step in range(100):
    obs, _, done, info = env.step(env.action_space.sample())
    print(obs)
    # env.render()

    x, task_id = obs["x"], obs["task_labels"]

    if 0 <= step < 10:
        assert env.length == starting_length and env.gravity == starting_gravity
        assert task_id == 0, step

    elif 10 <= step < 20:
        assert env.length == 0.1
        assert task_id == 1, step

    elif 20 <= step < 30:
        assert env.length == 0.2 and env.gravity == -12.0
        assert task_id == 2, step

    elif step >= 30:
        assert env.length == starting_length and env.gravity == 0.9
        assert task_id == 3, step

    if done:
        obs = env.reset()
        assert isinstance(obs, dict)

env.close()
