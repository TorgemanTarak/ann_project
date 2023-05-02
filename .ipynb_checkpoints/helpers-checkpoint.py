import gym
import numpy as np
import torch
from tqdm import tqdm
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
    
    
class RandomAgent:
    def __init__(self, env):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        
    def compute_action(self, state):
        return np.random.uniform(-1, 1, size=(self.action_size, ))
    
    
    
class Buffer:    
    def __init__(self, capacity=10000,state_dim=3,action_dim=1):
        
        self.capacity = capacity
        
       
        
        self.num_samples = 0

        # followong SARSA we need to store: state ,action ,reward ,next state ,next action
        self.state_tape = torch.zeros((self.capacity, state_dim))
        self.action_tape = torch.zeros((self.capacity, action_dim))
        self.reward_tape = torch.zeros((self.capacity, 1))
        self.next_state_tape = torch.zeros((self.capacity, state_dim))
        self.next_action_tape = torch.zeros((self.capacity, action_dim))
        
    def add(self, sample):
       # sample = (state, action, reward, next_state, next_action)
        #to replace
        index = self.num_samples % self.capacity

        self.state_tape[index] = torch.from_numpy(sample[0])
        self.action_tape[index] = sample[1]
        self.reward_tape[index] = sample[2]
        self.next_state_tape[index] = torch.from_numpy(sample[3])
        self.next_action_tape[index] = sample[4]
        self.num_samples += 1
    def sample_batch(self,batch_size):
        record_range = min(self.num_samples, self.capacity)
        indices = torch.randint(0,record_range,size=(batch_size,))
        return indices



        
class Qnetwork(nn.Module):
    def __init__(self,state_dim=3,action_dim=1,hidden_dim=32):
        super().__init__()
        self.input_shape = state_dim + action_dim
        self.fc1 = nn.Linear(self.input_shape,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,1)
        self.activation=nn.ReLU()
    def forward(self,x):
        output = self.fc1(x)
        output = self.activation(output)
        output = self.fc2(output)
        output = self.activation(output)
        output = self.fc3(output)
        return output
            
def update_average(mparam, tparam,beta):
        
        return tparam * (1-beta) + (beta) * mparam

def EMA(model,target,beta):
    for current_params, ma_params in zip(
            model.parameters(), target.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = update_average(up_weight, old_weight,beta)

            


class PolicyNetwork(nn.Module):
    def __init__(self,state_dim=3,hidden_dim=32):
        super().__init__()
        self.input_shape = state_dim
        self.fc1 = nn.Linear(self.input_shape,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,1)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.Tanh()
        #we dont want to saturate the tanh and kill the gradient too early
        #torch.nn.init.uniform_(self.fc3.weight, -0.005, 0.005)
    def forward(self,x):
        output = self.fc1(x)
        output = self.activation1(output)
        output = self.fc2(output)
        output = self.activation1(output)
        output = self.fc3(output)
        output = self.activation2(output)
        return output
    
class GaussianActionNoise():
    def __init__(self,sigma):
        self.sigma=sigma
        
    def get_noisy_action(self,action):
        noise = torch.normal(0.0,self.sigma,action.shape).to(device)
        action = torch.clamp(noise + action,-1,1)
        return action
    
class OUActionNoise():
    def __init__(self,sigma,theta=0.15):
        self.sigma=sigma
        self.prev=0.0
        self.theta=theta
        
    def evolve_state(self,action):
        noise = (1 - self.theta) * self.prev + torch.normal(0.0,self.sigma,action.shape).to(device)
        self.prev = noise
        action = torch.clamp(noise + action,-1,1)
        return action
    
    def get_noisy_action(self,action):
        noise = torch.normal(0.0,self.sigma,action.shape).to(device)
        action = torch.clamp(noise + action,-1,1)
        return action
    
class DDPGAgent():
    def __init__(self,model,sigma,vanilla_noise=True,theta=0):
        self.model=model
        self.vanilla_noise=vanilla_noise
        
        self.noise=OUActionNoise(sigma)
    def compute_action(self,state, deterministic=True):
        action = self.model(state)
        if deterministic:
            return action
        else:
            if self.vanilla_noise:
                return self.noise.get_noisy_action(action)
            else:
                return self.noise.evolve_state(action)
            
            