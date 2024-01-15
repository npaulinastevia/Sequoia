import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gym

class CNNQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self._calculate_fc_input_size(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _calculate_fc_input_size(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        x = self._convolutional_layers(dummy_input)
        return x.view(1, -1).size(1)

    def _convolutional_layers(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return torch.relu(self.conv3(x))

    def forward(self, x):
        x = self._convolutional_layers(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.age = 0

    def add(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.age % self.capacity] = transition
        self.age += 1

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

def huber_loss(y_pred, y_true, delta=1.0):
    error = y_true - y_pred
    cond = torch.abs(error) < delta
    loss = torch.where(cond, 0.5 * error**2, delta * (torch.abs(error) - 0.5 * delta))
    return loss.mean()

def preprocess_state(state):
    state = np.array(state, dtype=np.float32)
    state = np.moveaxis(state, 2, 0)
    return torch.tensor(state)

def deep_q_learning(env, frame_limit, theta, alpha, beta, gamma, steps, k, EQ):
    state_size = env.observation_space.shape
    action_size = env.action_space.n
    q_network = CNNQNetwork(state_size, action_size)
    q_target_network = CNNQNetwork(state_size, action_size)
    q_target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=alpha)

    replay_buffer = ExperienceReplayBuffer(capacity=50000)

    for frame in range(frame_limit):
        state = preprocess_state(env.reset())
        episode_done = False

        while not episode_done:
            # Exploration-exploitation strategy
            if random.random() <= epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_network(state.unsqueeze(0))
                    action = torch.argmax(q_values).item()

            next_state, reward, episode_done, _ = env.step(action)
            next_state = preprocess_state(next_state)

            # Store transition in replay buffer
            replay_buffer.add((state, action, reward, next_state))

            # Sample from replay buffer
            batch = replay_buffer.sample(steps)

            # Reptile meta-update
            for i in range(steps):
                theta_temp = theta.clone()
                for j in range(k):
                    s, a, r, s_next = batch[i]
                    s, s_next = s.unsqueeze(0), s_next.unsqueeze(0)
                    q_values = q_network(s)
                    q_target_values = q_target_network(s_next)
                    y = r + gamma * torch.max(q_target_values) if not episode_done else r
                    loss = huber_loss(q_values[0, a], y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    theta_temp -= alpha * theta.grad

                # Reptile meta-update
                theta = theta + beta * (theta_temp - theta)

            # Update target network every EQ episodes
            if frame % EQ == 0:
                q_target_network.load_state_dict(q_network.state_dict())

            state = next_state

    return theta, replay_buffer

# Example usage:
env = gym.make('CartPole-v1')
frame_limit = 10000
theta = torch.rand((1,), requires_grad=True)
alpha = 0.001
beta = 0.001
gamma = 0.99
steps = 32
k = 16
EQ = 100
epsilon = 0.1  # Adjust exploration-exploitation strategy as needed

theta_final, replay_buffer = deep_q_learning(env, frame_limit, theta, alpha, beta, gamma, steps, k, EQ)
