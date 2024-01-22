import gym
from dyna_gym.envs import nscartpole_v0, NSCartPoleV0, NSCartPoleV1,NSCartPoleV2
env = NSCartPoleV1()#CartPole-v1
se_actions=2
frame_limit = 25
alpha = 0.001
beta = 0.001
gamma = 0.99
steps = 64
k = 32
EQ = 100
  # Adjust exploration-exploitation strategy as needed
frame_idx = 0
EPSILON_DECAY_LAST_FRAME = 1000
EPSILON_START = 0.9
EPSILON_FINAL = 0.05
epsilon = EPSILON_START
