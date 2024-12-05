#Copied the implementation from here - https://github.com/seungeunrho/minimalRL/blob/master/REINFORCE.py

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
class ValueNetwork(nn.Module):
    '''
    Value network V(s_t) = E[G_t | s_t] to use as a baseline in the reinforce
    update. This a Neural Net with 1 hidden layer
    '''

    def __init__(self, num_inputs, hidden_size):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, state):

        x = F.relu(self.linear1(state))
        x = self.linear2(x)

        return x

def main():

    # Hyperparameters
    learning_rate_pi = 0.0005
    learning_rate_v = 0.001
    gamma = 0.99
    n_episodes = 1000
    env = gym.make('HalfCheetah-v5')

    dimS = env.observation_space.shape[0]
    dimA = env.action_space.shape[0]
    
    pi = Policy(dimS, dimA, hidden_size=32).to(device)
    valuenet = ValueNetwork(dimS, hidden_size=64).to(device)
    pi_optimizer = optim.Adam(pi.parameters(), lr=learning_rate_pi)
    v_optimizer = optim.Adam(valuenet.parameters(), lr=learning_rate_v)

    score = 0.0
    print_interval = 20

    for n_epi in range(n_episodes):
        s, x_pos = env.reset()
        
        done = False
        truncated = False
        data = []
        while not done and not truncated: 
            mean, std = pi(torch.from_numpy(s).float().to(device))
            normal = Normal(mean, std) # create normal distribution
            a = normal.sample().tanh() # sample action
            s_prime, r, done, truncated, info = env.step(a.cpu().numpy())
            v = valuenet(torch.from_numpy(s).float().to(device))
            log_prob = normal.log_prob(a).sum()
            data.append((r, v, log_prob))
            s = s_prime
            score += r
    
        R = 0
        returns = []
        for r, _, _ in data[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float().to(device)
        
        # gradient descent on value network
        values = [x[1] for x in data]
        values = torch.stack(values).squeeze(1).to(device)
        v_loss = F.mse_loss(values, returns)
        v_optimizer.zero_grad()
        v_loss.backward()
        v_optimizer.step()     

        # gradient descent on policy network 
        advantage = []
        for value, R in zip(values, returns):
            advantage.append(R - value)
        advantage = torch.tensor(advantage).to(device)
        log_probs = [x[2] for x in data]
        log_probs = torch.stack(log_probs)

        pi_loss = torch.tensor(0).float().to(device) # caluclate policy loss
        for log_prob, adv in zip(log_probs, advantage):
            pi_loss += - log_prob * adv
        pi_optimizer.zero_grad()
        pi_loss.backward()
        pi_optimizer.step()

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