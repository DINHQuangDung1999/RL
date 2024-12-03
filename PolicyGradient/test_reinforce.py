from comet_ml import Experiment
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
import json
from reinforce.REINFORCE_continuous import REINFORCE
from reinforce.REINFORCE_discrete import REINFORCE_discrete
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import os
import gymnasium as gym

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def evaluate_policy(policy, env, eval_episodes = 10):
    '''
        function to return the average reward of the policy over 10 runs
    '''
    avg_reward = 0.0
    for _ in range(eval_episodes):
        if action_space == 'continuous':
            obs, x_pos = env.reset()
        else:
            obs, _ = env.reset()
            tmp = np.zeros(state_dim)
            tmp[obs] = 1
            obs = tmp
        terminated = False
        truncated = False
        
        while (not terminated and not truncated):

            action, prob = policy.select_action(np.array(obs))
            
            next_state, reward, terminated, truncated, current_pos = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    return avg_reward


# env_name = 'HalfCheetah-v5'
# action_space = 'continuous' 
env_name = "FrozenLake-v1"
action_space = 'discrete'
seed = 28051999
torch.manual_seed(seed)
np.random.seed(seed)
baseline = False
lr_pi = 1e-3
lr_vf = 1e-3
gamma = 0.99
max_episodes = 20000
hidden_size = 64 # Often default = 256 # follow different logic depending on action space of env


# create env
env = gym.make(env_name)

if action_space == "continuous":
    # get env info
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = (env.action_space.high)
    min_action = (env.action_space.low)
    print("=======================")
    print(f"Env: {env_name} - {action_space} action space.")
    print("dim(A): {0}, dim(S): {1}, max_action: {2}, min_action: {3}".format(action_dim,\
                                                state_dim,max_action,min_action))
    # create policy
    policy = REINFORCE(state_dim, hidden_size, action_dim, baseline = baseline, lr_pi=lr_pi, lr_vf=lr_vf, gamma=gamma)

elif action_space == "discrete":
    # get env info
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    print("=======================")
    print(f"Env: {env_name} - {action_space} action space.")
    print("dim(A): {0}, dim(S): {1}.".format(action_dim,state_dim))

    # create policy
    policy = REINFORCE_discrete(state_dim, hidden_size, action_dim, baseline = baseline, lr_pi=lr_pi, lr_vf=lr_vf, gamma=gamma)

else:
    raise NotImplementedError

training_hist = {'ValueFunctionLoss': np.zeros(max_episodes), 
                    'PolicyLoss': np.zeros(max_episodes), 
                    'EpisodeReward': np.zeros(max_episodes)}
# start of experiment: Keep looping until desired amount of episodes reached

total_episodes = 0 # keep track of amount of episodes that we have done
print("=======================")
while total_episodes < max_episodes:

    if action_space == 'continuous':
        obs, x_pos = env.reset()
    else:
        obs, _ = env.reset()
        tmp = np.zeros(state_dim)
        tmp[obs] = 1
        obs = tmp
    terminated = False
    truncated = False
    trajectory = [] # trajectory info for reinforce update
    episode_reward = 0 # keep track of rewards per episode

    while (not terminated and not truncated):
        action, prob = policy.select_action(np.array(obs))
        next_state, reward, terminated, truncated, current_pos = env.step(action)
        trajectory.append([np.array(obs), action, prob, reward, next_state, terminated])
        if action_space == 'continuous':
            obs = next_state
        else:
            tmp = np.zeros(state_dim)
            tmp[next_state] = 1
            obs = tmp
        episode_reward += reward

    if baseline:
        policy_loss, value_loss = policy.train(trajectory)
        training_hist['ValueFunctionLoss'][total_episodes] = value_loss
    else:
        policy_loss = policy.train(trajectory)
    training_hist['PolicyLoss'][total_episodes] = policy_loss
    training_hist['EpisodeReward'][total_episodes] = episode_reward

    total_episodes += 1

    if total_episodes % 10 == 0:
        avg_rewards = evaluate_policy(policy,env)
        
        print(f"Episodes[{total_episodes}/{max_episodes}] | AvgReward: {round(avg_rewards, 2)}.")  
    env.close()
torch.save(policy.policy.state_dict(), f'./checkpoints/{env_name}-{max_episodes}.pt')

print("=======================")
for i in range(10):
    # Render the policy
    env = gym.make(env_name, render_mode = 'human')
    if action_space == 'continuous':
        obs, x_pos = env.reset()
    else:
        obs, _ = env.reset()
        tmp = np.zeros(state_dim)
        tmp[obs] = 1
        obs = tmp
    terminated = False
    truncated = False
    while (not terminated and not truncated):
        action, prob = policy.select_action(np.array(obs))
        next_state, reward, terminated, truncated, current_pos = env.step(action)
        if action_space == 'continuous':
            obs = next_state
        else:
            tmp = np.zeros(state_dim)
            tmp[next_state] = 1
            obs = tmp
        print('Prob:{}'.format(torch.exp(prob)))


