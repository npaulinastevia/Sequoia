import torch
import torch.nn as nn
import torch.optim as optim
import random
import gym
import numpy as np

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the Huber loss
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    cond = torch.abs(error) < delta
    loss = torch.where(cond, 0.5 * error**2, delta * (torch.abs(error) - 0.5 * delta))
    return loss.mean()


# Function to perform experience replay
def sample_batch(buffer, batch_size):
    batch = random.sample(buffer, batch_size)
    states, actions, rewards, next_states = zip(*batch)
    return (
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(rewards, dtype=torch.float32),
        torch.tensor(next_states, dtype=torch.float32),
    )


# DQN-MER implementation
def dqn_mer(env, frame_limit, theta, alpha, beta, gamma, steps, k, eq):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    Q = QNetwork(state_size, action_size)
    Q_hat = QNetwork(state_size, action_size)
    Q_hat.load_state_dict(Q.state_dict())
    optimizer = optim.Adam(Q.parameters(), lr=alpha)

    replay_buffer = []
    replay_buffer_age = 0

    for frame in range(frame_limit):
        state = env.reset()
        episode_done = False

        while not episode_done:
            epsilon = 0.1  # Replace with your exploration strategy
            action = (
                random.randint(0, action_size - 1)
                if random.uniform(0, 1) < epsilon
                else torch.argmax(Q(torch.tensor(state, dtype=torch.float32))).item()
            )

            next_state, reward, episode_done, _ = env.step(action)

            replay_buffer.append((state, action, reward, next_state))
            replay_buffer_age += 1

            if replay_buffer_age >= steps:
                for i in range(steps):
                    Q_temp = QNetwork(state_size, action_size)
                    Q_temp.load_state_dict(Q.state_dict())
                    theta_a_0 = Q.state_dict()
                    for j in range(k):
                        batch = sample_batch(replay_buffer, steps)
                        states, actions, rewards, next_states = batch
                        with torch.no_grad():
                            y = rewards + gamma * Q_hat(next_states).max(dim=1)[0]
                            y[episode_done] = rewards[episode_done]

                        loss = huber_loss(Q(states).gather(1, actions.unsqueeze(1)), y.unsqueeze(1))
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        theta_w_i_j = Q.state_dict()

                        for name in Q_temp.state_dict():
                            theta_w_i_j[name] = theta_w_i_j[name] - alpha * theta_w_i_j[name].grad

                    theta_w_i_0 = Q_temp.state_dict()
                    for name in theta_w_i_0:
                        theta_a_0[name] = theta_w_i_0[name] + beta * (theta_w_i_j[name] - theta_w_i_0[name])

                    Q.load_state_dict(theta_a_0)

                for name in Q_hat.state_dict():
                    theta[name] = theta[name] + gamma * (theta_w_i_j[name] - theta[name])

                replay_buffer_age = 0

        if frame % eq == 0:
            Q_hat.load_state_dict(Q.state_dict())

    return theta, replay_buffer

# Example usage:
env = gym.make('CartPole-v1')
theta_init = QNetwork(env.observation_space.shape[0], env.action_space.n).state_dict()
theta_final, replay_buffer_final = dqn_mer(env, frame_limit=10000, theta=theta_init,
                                           alpha=0.001, beta=0.001, gamma=0.99, steps=32, k=5, eq=10)
