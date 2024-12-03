# import dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gym
from collections import deque
import random
import matplotlib.pyplot as plt 

class PolicyNet(nn.Module):
    def __init__(self, dimS):
        super(PolicyNet, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(dimS, 10)
        self.fc2 = nn.Linear(10, 4)
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x


class ValueNet(nn.Module):
    def __init__(self, state_dim, n_hidden):
        super(ValueNet, self).__init__()
        self.linear1 = nn.Linear(state_dim, n_hidden)
        self.linear2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        V = self.linear2(out)
        return V

def onehot(i, n):
    oh = np.zeros(n)
    oh[i] = 1
    return oh

gamma=0.99
learning_rate=0.0005
def main():
    env = gym.make("FrozenLake-v1") # sample toy environment
    env.is_slippery()
    dimS = env.observation_space.n
    policy = PolicyNet(dimS=dimS)
    score = 0.0
    print_interval = 20
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    for n_ep in range(20000):
        state, _ = env.reset()
        state = onehot(state, dimS)
        done = False
        data = []
        while not done:
            probs = policy(torch.from_numpy(state).float())
            policy_prob_dist = Categorical(probs)
            action = policy_prob_dist.sample()
            new_state, reward, done, truncated, info = env.step(action.item())
            new_state = onehot(new_state, dimS)
            data.append((reward,probs[action]))
            state = new_state
            score += reward
        R = 0
        weight_b4 = policy.fc1.weight.clone(), policy.fc2.weight.clone()
        policy_optimizer.zero_grad()
        loss = torch.tensor(0).float()
        for r, prob in data[::-1]:
            R = r + gamma * R
            loss = loss-torch.log(prob) * R
        loss.backward()
        policy_optimizer.step()
        weight_af = policy.fc1.weight, policy.fc2.weight
        if all(torch.sum((weight_b4[i]-weight_af[i]).abs()) == 0 for i in range(len(weight_b4))) and loss != 0:
            print(loss)
            breakpoint()
        if n_ep % print_interval == 0 and n_ep != 0:
            print("# of episode :{}, avg score : {}".format(n_ep, score / print_interval))
            score = 0.0

    env.close()

    print("=======================")
    env = gym.make("FrozenLake-v1", render_mode = 'human')
    for i in range(10):
        state, _ = env.reset()
        state = onehot(state, dimS)
        done = False
        while not done:
            # recieve action probabilities from policy function
            probs = policy(torch.from_numpy(state).float())
            # sample an action from the policy distribution
            policy_prob_dist = Categorical(probs)
            action = policy_prob_dist.sample()

            # take that action in the environment
            new_state, reward, done, truncated, info = env.step(action.item())
            new_state = onehot(new_state, dimS)
            state = new_state

if __name__ == '__main__':
    main()