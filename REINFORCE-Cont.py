#Copied the implementation from here - https://github.com/seungeunrho/minimalRL/blob/master/REINFORCE.py

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class Policy(nn.Module):
    '''
    Gaussian policy that consists of a neural network with 1 hidden layer that
    outputs mean and log std dev (the params) of a gaussian policy
    '''

    LOG_SIG_MAX = 2
    LOG_SIG_MIN = -2

    def __init__(self, dimS, dimA, hidden_size):

        super(Policy, self).__init__()

        self.linear = nn.Linear(dimS, hidden_size)
        self.mean = nn.Linear(hidden_size, dimA)
        self.log_std = nn.Linear(hidden_size, dimA)

    def forward(self, inputs):
        
        x = F.relu(self.linear(inputs))

        mean = self.mean(x)
        log_std = self.log_std(x) # if more than one action this will give you the diagonal elements of a diagonal covariance matrix
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX) # We limit the variance by forcing within a range of -2,2
        std = log_std.exp()

        return mean, std

def main():

    # Hyperparameters
    learning_rate = 0.001
    gamma = 0.99
    n_episodes = 3000
    env = gym.make('HalfCheetah-v5')
    dimS = env.observation_space.shape[0]
    dimA = env.action_space.shape[0]
    
    pi = Policy(dimS, dimA, hidden_size=10)
    optimizer = optim.Adam(pi.parameters(), lr=learning_rate)

    score = 0.0
    print_interval = 20

    for n_epi in range(n_episodes):
        s, x_pos = env.reset()
        
        done = False
        truncated = False
        data = []
        while not done and not truncated: 
            mean, std = pi(torch.from_numpy(s).float())
            # create normal distribution
            normal = Normal(mean, std)
            # sample action
            a = normal.sample().tanh()
            s_prime, r, done, truncated, info = env.step(a.numpy())
            log_prob = normal.log_prob(a).sum()
            data.append((r, log_prob))
            s = s_prime
            score += r

        # pi.train_net()
        R = 0
        optimizer.zero_grad()
        loss = torch.tensor(0).float()
        for r, log_prob in data[::-1]:
            R = r + gamma * R
            loss = loss-log_prob * R
        loss.backward()
        optimizer.step()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.2f}".format(n_epi, score / print_interval))
            score = 0.0
    env.close()

    print("Testing")
    env = gym.make('HalfCheetah-v5', render_mode="human")

    s, x_pos = env.reset()
    done = False
    truncated = False
    while not done and not truncated: 
        mean, std = pi(torch.from_numpy(s).float())
        normal = Normal(mean, std) # create normal distribution
        a = normal.sample().tanh() # sample action
        s_prime, r, done, truncated, info = env.step(a.numpy())
        s = s_prime
    env.close()

if __name__ == '__main__':
    main()