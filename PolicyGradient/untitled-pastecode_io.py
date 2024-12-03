#Copied the implementation from here - https://github.com/seungeunrho/minimalRL/blob/master/REINFORCE.py

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
# Hyperparameters
learning_rate = 0.0005
gamma = 0.99


class Policy(nn.Module):
    def __init__(self, dimS):
        super(Policy, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(dimS, 10)
        self.fc2 = nn.Linear(10, 4)
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x

    # def put_data(self, item):
    #     self.data.append(item)

    # def train_net(self):
    #     R = 0
    #     self.optimizer.zero_grad()
    #     loss = torch.tensor(0).float()
    #     for r, prob in self.data[::-1]:
    #         R = r + gamma * R
    #         loss = loss-torch.log(prob) * R
    #     loss.backward()
    #     self.optimizer.step()
    #     self.data = []

def onehot(i, n):
    oh = np.zeros(n)
    oh[i] = 1
    return oh

def main():
    env = gym.make('FrozenLake-v1', is_slippery=False)
    dimS = env.observation_space.n
    pi = Policy(dimS)
    score = 0.0
    print_interval = 20
    optimizer = optim.Adam(pi.parameters(), lr=learning_rate)
    for n_epi in range(20000):
        s, _ = env.reset()
        s = onehot(s, dimS)
        done = False
        data = []
        while not done:  # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, truncated, info = env.step(a.item())
            s_prime = onehot(s_prime, dimS)
            # pi.put_data((r, prob[a]))
            data.append((r, prob[a]))
            s = s_prime
            score += r

        # pi.train_net()
        R = 0
        optimizer.zero_grad()
        loss = torch.tensor(0).float()
        for r, prob in data[::-1]:
            R = r + gamma * R
            loss = loss-torch.log(prob) * R
        loss.backward()
        optimizer.step()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()

    print("Testing")
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")

    for n_epi in range(10):
        s, _ = env.reset()
        s = onehot(s, dimS)
        done = False

        while not done:  # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            s_prime, r, done, truncated, info = env.step(a.item())
            s_prime = onehot(s_prime, dimS)
            s = s_prime

if __name__ == '__main__':
    main()