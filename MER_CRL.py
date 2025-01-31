import collections
import math

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
        x=x.float()
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

import time
id_txt = str(int(time.time()))
#f = open('results_cart_testCRL' + id_txt + '.txt', 'a+')
#f.write('Environments,Algorithms,x1,reward,steps,time,episodes' + '\n')
#f.close()
class MER_CRL:

    def __init__(self,env, frame_limit,  alpha, beta, gamma, steps, k, EQ,iter):
        self.env, self.frame_limit, self.alpha, self.beta, self.gamma, self.steps, self.k, self.EQ=(
            env, frame_limit,  alpha, beta, gamma, steps, k, EQ)
        self.state_size = env.observation_space.shape
        self.action_size = env.action_space.n
        self.q_network = CNNQNetwork(self.state_size, self.action_size)
        self.q_target_network = CNNQNetwork(self.state_size, self.action_size)
        self.q_target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)
        self.env_name=config.env_name
        self.frame_idx=0
        import time
        self.start=time.time()
        self.iter=iter
        self.task_schedule_cartpole = {
            0: {"gravity": 9.8, "length": 0.5},
            #0: {"gravity": 10, "length": 0.2},
            #10: {"gravity": 100, "length": 1.2},
            #64: {"gravity": 10, "length": 0.2},
            #150: dict(length=0.1,gravity=9.8),
            #200: dict(length=0.2, gravity=-12.0),
            #250: dict(length=0.5,gravity=0.9),
        }

    def deep_q_learning_test(self):

        #theta = torch.rand((1,), requires_grad=True)

        filename = 'bug_log_RELINE'+str(self.iter)+str(self.env.action_space.n)+'.txt'
        f = open('bug_log_RELINE'+str(self.iter)+str(self.env.action_space.n)+'.txt', 'w')
        f.close()
        replay_buffer = ExperienceReplayBuffer(capacity=50000)
        flag_injected_bug_spotted = [False, False]
        self.q_network.load_state_dict(torch.load('./model_RELINE'+str(self.env_name)))
        self.q_target_network.load_state_dict(self.q_network.state_dict())
        for frame in range(100):
            if self.env_name == 'Cartpolev0':

                state = preprocess_state(self.env.reset()[0])
            else:
                state = preprocess_state(self.env.reset()[:4])
            episode_done = False

            while not episode_done:
                self.frame_idx+= 1
                # Exploration-exploitation strategy
                epsilon =  max(config.EPSILON_FINAL, config.EPSILON_START - self.frame_idx / config.EPSILON_DECAY_LAST_FRAME)
                if random.random() <= epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = self.q_network(torch.from_numpy(state).unsqueeze(0))
                        action = torch.argmax(q_values).item()

                if self.env_name == 'Cartpolev0':
                    next_state, reward, episode_done, _, _ = self.env.step(action)
                else:
                    next_state, reward, episode_done, _ = self.env.step(action)
                next_state=next_state[:4]
                f = open('results_cart_testCRL1706181975.txt', 'a+')
                f.write(str(self.env_name)+','+'DQN_CRL'+','+str(next_state[0])+','+str(reward)+','+str(self.frame_idx)+','+str(time.time()-self.start)+','+str(frame) + '\n')
                f.close()
                if -0.5 < next_state[0] < -0.45 and not flag_injected_bug_spotted[0]:
                    reward += 50
                    flag_injected_bug_spotted[0] = True
                if 0.45 < next_state[0] < 0.5 and not flag_injected_bug_spotted[1]:
                    reward += 50
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
                import copy
                before = copy.deepcopy(self.q_network.state_dict())
                for i in range(self.steps):
                    states, actions, rewards, dones, next_states = batch
                    Q_temp = CNNQNetwork(self.state_size, self.action_size)
                    Q_temp.load_state_dict(self.q_network.state_dict())
                    weights_before = copy.deepcopy(self.q_network.state_dict())
                    #theta_a_0 = self.q_network.state_dict()

                    for j in range(self.k):

                        s, a, r,_, s_next = states[i], actions[i], rewards[i], dones[i], next_states[i]#batch[i]
                        s, s_next=torch.from_numpy(s).requires_grad_(True),torch.from_numpy(s_next).requires_grad_(True)
                        s, s_next = s.unsqueeze(0), s_next.unsqueeze(0)
                        q_values = self.q_network(s)
                        q_target_values = self.q_target_network(s_next)
                        y = r + self.gamma * torch.max(q_target_values) if not episode_done else torch.tensor(r,requires_grad=True)

                        loss = nn.SmoothL1Loss()(q_values[0, a], y)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        #grad_dict = {k: v.grad for k, v in zip(q_network.state_dict(), q_network.parameters())}
                        #print(grad_dict)
                        #print('eeeeeeeeeeeeeeeeeee')
                        #theta_w_i_j = self.q_network.state_dict()

                        #for name in Q_temp.state_dict():

                        #    if theta_w_i_j[name].grad:
                        #        theta_w_i_j[name] = theta_w_i_j[name] - self.alpha * theta_w_i_j[name].grad
                    weights_after = self.q_network.state_dict()
                    self.q_network.load_state_dict(
                        {name: weights_before[name] + ((weights_after[name] - weights_before[name]) * self.beta) for
                         name in weights_before})
                    # Reptile meta-update
                    #theta_w_i_0 = Q_temp.state_dict()
                    #for name in theta_w_i_0:
                     #   theta_a_0[name] = theta_w_i_0[name] + self.beta * (theta_w_i_j[name] - theta_w_i_0[name])

                    #self.q_network.load_state_dict(theta_a_0)

                #for name in self.q_target_network.state_dict():
                #    self.theta[name] = self.theta[name] + self.gamma * (theta_w_i_j[name] - self.theta[name])
                after = self.q_network.state_dict()

                # Across batch Reptile meta-update:
                self.q_network.load_state_dict(
                    {name: before[name] + ((after[name] - before[name]) * self.gamma) for name in before})

                # Reptile meta-update

            if frame % self.EQ == 0:
                self.q_target_network.load_state_dict(self.q_network.state_dict())
                # Update target network every EQ episodes
            f = open('bug_log_RELINE'+str(self.iter)+str(self.env.action_space.n)+'.txt', 'a+')
            if flag_injected_bug_spotted[0]:
                f.write('BUG1 ')
            if flag_injected_bug_spotted[1]:
                f.write('BUG2 ')
            f.write('\n')
            f.close()
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




        return None, None
    def deep_q_learning(self):

        #theta = torch.rand((1,), requires_grad=True)
        self.theta = CNNQNetwork(self.state_size, self.action_size).state_dict()

        replay_buffer = ExperienceReplayBuffer(capacity=50000)
        task_id=0
        total_step=0
        flag_injected_bug_spotted = [False, False]
        for frame in range(self.frame_limit):
            print(frame)
            if self.env_name=='Cartpolev0':

                state = preprocess_state(self.env.reset()[0])
            else:


                state = preprocess_state(self.env.reset()[:4])

            episode_done = False

            while not episode_done:
                # if task_id == 0:
                #     self.env.gravity = self.task_schedule_cartpole[task_id]['gravity']
                #     self.env.length = self.task_schedule_cartpole[task_id]['length']
                #     state = self.env.reset()
                #     task_id = task_id + 1
                # else:
                #     if task_id < len(list(self.task_schedule_cartpole.keys())):
                #
                #         if total_step == sorted(list(self.task_schedule_cartpole.keys()))[task_id]:
                #             self.env.gravity = self.task_schedule_cartpole[total_step]['gravity']
                #             self.env.length = self.task_schedule_cartpole[total_step]['length']
                #             state = self.env.reset()
                #             task_id = task_id + 1
                self.frame_idx += 1
                # Exploration-exploitation strategy
                #epsilon = max(config.EPSILON_FINAL, config.EPSILON_START - self.frame_idx / config.EPSILON_DECAY_LAST_FRAME)
                epsilon =config.EPSILON_FINAL + (config.EPSILON_START - config.EPSILON_FINAL) * \
                math.exp(-1. * self.frame_idx / config.EPSILON_DECAY_LAST_FRAME)
                if random.random() <= epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad():
                        q_values = self.q_network(torch.from_numpy(state).unsqueeze(0))
                        action = torch.argmax(q_values).item()
                import time
                if self.env_name == 'Cartpolev0':
                    next_state, reward, episode_done, _,_ = self.env.step(action)
                else:
                    next_state, reward, episode_done, _ = self.env.step(action)
                next_state = next_state[:4]
                f = open('results_cart_trainCRL1706180927.txt', 'a+')
                f.write(str(self.env_name)+','+'DQN_CRL'+','+str(next_state[0])+','+str(reward)+','+str(self.frame_idx)+','+str(time.time()-self.start)+','+str(frame) + '\n')
                f.close()
                #self.file = 'results_cart_trainCRL' + self.id_txt + '.txt'
                if -0.5 < next_state[0] < -0.45 and not flag_injected_bug_spotted[0]:
                    reward += 50
                    flag_injected_bug_spotted[0] = True
                if 0.45 < next_state[0] < 0.5 and not flag_injected_bug_spotted[1]:
                    reward += 50
                    flag_injected_bug_spotted[1] = True
                next_state = preprocess_state(next_state)

                # Store transition in replay buffer
                exp = Experience(state, action, reward, episode_done, next_state)
                replay_buffer.append(exp)
                state = next_state
                total_step = total_step + 1
                if len(replay_buffer) < self.steps:
                    continue
                # Sample from replay buffer
                batch = replay_buffer.sample(self.steps)

                # Reptile meta-update
                import copy
                before = copy.deepcopy(self.q_network.state_dict())
                for i in range(self.steps):
                    states, actions, rewards, dones, next_states = batch
                    Q_temp = CNNQNetwork(self.state_size, self.action_size)
                    Q_temp.load_state_dict(self.q_network.state_dict())
                    weights_before = copy.deepcopy(self.q_network.state_dict())
                    #theta_a_0 = self.q_network.state_dict()

                    for j in range(self.k):

                        s, a, r,_, s_next = states[i], actions[i], rewards[i], dones[i], next_states[i]#batch[i]
                        s, s_next=torch.from_numpy(s).requires_grad_(True),torch.from_numpy(s_next).requires_grad_(True)
                        s, s_next = s.unsqueeze(0), s_next.unsqueeze(0)
                        q_values = self.q_network(s)
                        q_target_values = self.q_target_network(s_next)
                        y = r + self.gamma * torch.max(q_target_values) if not episode_done else torch.tensor(r,requires_grad=True)

                        loss = nn.SmoothL1Loss()(q_values[0, a], y)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                    weights_after = self.q_network.state_dict()
                    self.q_network.load_state_dict(
                        {name: weights_before[name] + ((weights_after[name] - weights_before[name]) * self.beta) for
                         name in weights_before})
                    # Reptile meta-update

                after = self.q_network.state_dict()

                # Across batch Reptile meta-update:
                self.q_network.load_state_dict(
                    {name: before[name] + ((after[name] - before[name]) * self.gamma) for name in before})

                # Update target network every EQ episodes
            flag_injected_bug_spotted = [False, False]
            if frame % self.EQ == 0:
                self.q_target_network.load_state_dict(self.q_network.state_dict())

        torch.save(self.q_network.state_dict(), './model_RELINE'+str(self.env_name))

        return None, None

# Example usage:
if __name__ == '__main__':
    for m in range(1):
        Meta_agent=MER_CRL(env=config.env, frame_limit=config.frame_limit,  alpha=config.alpha, beta=config.beta, gamma=config.gamma, steps=config.steps, k=config.k, EQ=config.EQ,iter=m)
        theta_final, replay_buffer = Meta_agent.deep_q_learning()
        theta_final, replay_buffer = Meta_agent.deep_q_learning_test()

