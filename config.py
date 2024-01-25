import gym
from nscartpole_v0 import NSCartPoleV0
from nscartpole_v2 import NSCartPoleV2
from nscartpole_v1 import NSCartPoleV1
env = NSCartPoleV1()#gym.make('CartPole-v0')
env_name='NSCartpolev1'
frame_limit = 100
alpha = 1e-4
beta = 0.001
gamma = 0.99
steps = 16
k = 16
EQ = 100
  # Adjust exploration-exploitation strategy as needed
frame_idx = 0
EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
epsilon = EPSILON_START
