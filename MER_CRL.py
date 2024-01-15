import collections
import config
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import gym
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])



class CNNQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CNNQNetwork, self).__init__()
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self._calculate_fc_input_size(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)


    def _calculate_fc_input_size(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        if len(self.input_shape) > 2:
            x = self._convolutional_layers(dummy_input)
            return x.view(1, -1).size(1)
        else:
            return input_shape[0]

    def _convolutional_layers(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return torch.relu(self.conv3(x))

    def forward(self, x):
        if len(self.input_shape)>2:
            x = self._convolutional_layers(x)
            x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class ExperienceReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    # add experience
    def append(self, experience):
        self.buffer.append(experience)

    # provide a random batch of the experience
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
               np.array(dones, dtype=np.uint8), np.array(next_states)
class ExperienceReplayBuffer1:
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
    state = np.array(state)
    #state = np.moveaxis(state, 2, 0)
    return state

class MER_CRL:

    def __init__(self,env, frame_limit,  alpha, beta, gamma, steps, k, EQ):
        self.env, self.frame_limit, self.alpha, self.beta, self.gamma, self.steps, self.k, self.EQ=(
            env, frame_limit,  alpha, beta, gamma, steps, k, EQ)
        self.state_size = env.observation_space.shape
        self.action_size = env.action_space.n
        self.q_network = CNNQNetwork(self.state_size, self.action_size)
        self.q_target_network = CNNQNetwork(self.state_size, self.action_size)
        self.q_target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.frame_idx=0

    def deep_q_learning_test(self):

        #theta = torch.rand((1,), requires_grad=True)

        filename = 'injected_bugs_spotted_RELINE.txt'

        replay_buffer = ExperienceReplayBuffer(capacity=50000)
        flag_injected_bug_spotted = [False, False]

        for frame in range(50):
            state = preprocess_state(self.env.reset())
            episode_done = False

            while not episode_done:
                self.frame_idx+= 1
                # Exploration-exploitation strategy
                epsilon =  max(config.EPSILON_FINAL, config.EPSILON_START - self.frame_idx / config.EPSILON_DECAY_LAST_FRAME)
                if random.random() <= epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = self.q_network(state.unsqueeze(0))
                        action = torch.argmax(q_values).item()

                next_state, reward, episode_done, _ = self.env.step(action)
                if -0.5 < next_state[0] < -0.45 and not flag_injected_bug_spotted[0]:
                    reward += 50
                    filename.write('BUG1 ')
                    flag_injected_bug_spotted[0] = True
                if 0.45 < next_state[0] < 0.5 and not flag_injected_bug_spotted[1]:
                    reward += 50
                    filename.write('BUG1 ')
                    flag_injected_bug_spotted[1] = True
                next_state = preprocess_state(next_state)

                # Store transition in replay buffer
                exp = Experience(state, action, reward, episode_done, next_state)
                replay_buffer.append(exp)
                state = next_state
                if len(replay_buffer) < self.steps:
                    continue
                # Sample from replay buffer
                batch = replay_buffer.sample(self.steps)

                # Reptile meta-update

                for i in range(self.steps):
                    states, actions, rewards, dones, next_states = batch
                    Q_temp = CNNQNetwork(self.state_size, self.action_size)
                    Q_temp.load_state_dict(self.q_network.state_dict())
                    theta_a_0 = self.q_network.state_dict()

                    for j in range(self.k):

                        s, a, r,_, s_next = states[i], actions[i], rewards[i], dones[i], next_states[i]#batch[i]
                        s, s_next=torch.from_numpy(s).requires_grad_(True),torch.from_numpy(s_next).requires_grad_(True)
                        s, s_next = s.unsqueeze(0), s_next.unsqueeze(0)
                        q_values = self.q_network(s)
                        q_target_values = self.q_target_network(s_next)
                        y = r + self.gamma * torch.max(q_target_values) if not episode_done else r
                        loss = huber_loss(q_values[0, a], y)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        #grad_dict = {k: v.grad for k, v in zip(q_network.state_dict(), q_network.parameters())}
                        #print(grad_dict)
                        #print('eeeeeeeeeeeeeeeeeee')
                        theta_w_i_j = self.q_network.state_dict()

                        for name in Q_temp.state_dict():

                            if theta_w_i_j[name].grad:
                                theta_w_i_j[name] = theta_w_i_j[name] - self.alpha * theta_w_i_j[name].grad

                    # Reptile meta-update
                    theta_w_i_0 = Q_temp.state_dict()
                    for name in theta_w_i_0:
                        theta_a_0[name] = theta_w_i_0[name] + self.beta * (theta_w_i_j[name] - theta_w_i_0[name])

                    self.q_network.load_state_dict(theta_a_0)

                for name in self.q_target_network.state_dict():
                    self.theta[name] = self.theta[name] + self.gamma * (theta_w_i_j[name] - self.theta[name])

                # Update target network every EQ episodes
            flag_injected_bug_spotted = [False, False]
            lines = [line for line in open(filename, 'r')]
            lines_1k = lines[:1000]

            count_0bug = 0
            count_1bug = 0
            count_2bug = 0

            for line in lines_1k:
                if line.strip() == '':
                    count_0bug += 1
                elif len(line.strip().split()) == 1:
                    count_1bug += 1
                elif len(line.strip().split()) == 2:
                    count_2bug += 1
            print('Report injected bugs spotted:')
            print('0 injected bug spotted in %d episodes' % count_0bug)
            print('1 injected bug spotted in %d episodes' % count_1bug)
            print('2 injected bugs spotted in %d episodes' % count_2bug)
            print("\    /\ \n )  ( ')  meow!\n(  /  )\n \(__)|")
            if frame % self.EQ == 0:
                self.q_target_network.load_state_dict(self.q_network.state_dict())



        return self.theta, replay_buffer
    def deep_q_learning(self):

        #theta = torch.rand((1,), requires_grad=True)
        self.theta = CNNQNetwork(self.state_size, self.action_size).state_dict()

        replay_buffer = ExperienceReplayBuffer(capacity=50000)

        for frame in range(self.frame_limit):
            state = preprocess_state(self.env.reset())
            episode_done = False

            while not episode_done:
                self.frame_idx += 1
                # Exploration-exploitation strategy
                epsilon = max(config.EPSILON_FINAL, config.EPSILON_START - self.frame_idx / config.EPSILON_DECAY_LAST_FRAME)
                if random.random() <= epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = self.q_network(state.unsqueeze(0))
                        action = torch.argmax(q_values).item()

                next_state, reward, episode_done, _ = self.env.step(action)
                next_state = preprocess_state(next_state)

                # Store transition in replay buffer
                exp = Experience(state, action, reward, episode_done, next_state)
                replay_buffer.append(exp)
                state = next_state
                if len(replay_buffer) < self.steps:
                    continue
                # Sample from replay buffer
                batch = replay_buffer.sample(self.steps)

                # Reptile meta-update

                for i in range(self.steps):
                    states, actions, rewards, dones, next_states = batch
                    Q_temp = CNNQNetwork(self.state_size, self.action_size)
                    Q_temp.load_state_dict(self.q_network.state_dict())
                    theta_a_0 = self.q_network.state_dict()

                    for j in range(self.k):

                        s, a, r,_, s_next = states[i], actions[i], rewards[i], dones[i], next_states[i]#batch[i]
                        s, s_next=torch.from_numpy(s).requires_grad_(True),torch.from_numpy(s_next).requires_grad_(True)
                        s, s_next = s.unsqueeze(0), s_next.unsqueeze(0)
                        q_values = self.q_network(s)
                        q_target_values = self.q_target_network(s_next)
                        y = r + self.gamma * torch.max(q_target_values) if not episode_done else r
                        loss = huber_loss(q_values[0, a], y)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        #grad_dict = {k: v.grad for k, v in zip(q_network.state_dict(), q_network.parameters())}
                        #print(grad_dict)
                        #print('eeeeeeeeeeeeeeeeeee')
                        theta_w_i_j = self.q_network.state_dict()

                        for name in Q_temp.state_dict():

                            if theta_w_i_j[name].grad:
                                theta_w_i_j[name] = theta_w_i_j[name] - self.alpha * theta_w_i_j[name].grad

                    # Reptile meta-update
                    theta_w_i_0 = Q_temp.state_dict()
                    for name in theta_w_i_0:
                        theta_a_0[name] = theta_w_i_0[name] + self.beta * (theta_w_i_j[name] - theta_w_i_0[name])

                    self.q_network.load_state_dict(theta_a_0)

                for name in self.q_target_network.state_dict():
                    self.theta[name] = self.theta[name] + self.gamma * (theta_w_i_j[name] - self.theta[name])
                print('icici')
                # Update target network every EQ episodes
            if frame % self.EQ == 0:
                self.q_target_network.load_state_dict(self.q_network.state_dict())



        return self.theta, replay_buffer

# Example usage:
if __name__ == '__main__':
    Meta_agent=MER_CRL(env=config.env, frame_limit=config.frame_limit,  alpha=config.alpha, beta=config.beta, gamma=config.gamma, steps=config.steps, k=config.k, EQ=config.EQ)
    theta_final, replay_buffer = Meta_agent.deep_q_learning()
    theta_final, replay_buffer = Meta_agent.deep_q_learning_test()

