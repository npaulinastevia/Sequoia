import gym
env = gym.make('CartPole-v1')
frame_limit = 3
alpha = 0.001
beta = 0.001
gamma = 0.99
steps = 32
k = 16
EQ = 100
  # Adjust exploration-exploitation strategy as needed
frame_idx = 0
EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01
epsilon = EPSILON_START
