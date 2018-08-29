'''Actor and Critic for a Deep Determenistic Policy Gradients'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

EPS = 0.003


class Actor(nn.Module):
    '''Actor tries to predict a best possible next ection'''
    def __init__(self, state_size, action_size, action_low, action_high, use_gpu):
        '''
        Initializes variables
        
        :param state_size: Integer, state size of the environment
        :param action_size: Integer, size of possible actions
        :param action_low: Numpy array of lowest values for the actions
        :param action_high: Numpy array of highest values for the actions
        :param use_gpu: True if PyTorch should calculate on GPU
        '''
        super(Actor, self).__init__()

        self.action_range = Variable(torch.from_numpy(action_high - action_low).float())
        if use_gpu:
            self.action_range = self.action_range.cuda()

        self.fc1 = nn.Linear(state_size, 64)
        nn.init.xavier_uniform(self.fc1.weight)

        self.fc2 = nn.Linear(64, 64)
        nn.init.xavier_uniform(self.fc2.weight)

        self.fc3 = nn.Linear(64, action_size)
        nn.init.uniform(self.fc3.weight, -EPS, EPS)

    def forward(self, state):
        '''
        Forward propogation

        :param state: PyTorch Variable, environment state
        :return: desired action vector as PyTorch Tensor
        '''
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        raw_action = F.tanh(self.fc3(x))
        action = raw_action * self.action_range * 0.5

        return action


class Critic(nn.Module):
    '''Critic tries to optimize the actor predictions'''
    def __init__(self, state_size, action_size):
        '''
        Initializes variables
        
        :param state_size: Integer, state size of the environment
        :param action_size: Integer, size of possible actions
        '''
        super(Critic, self).__init__()

        self.fcs1 = nn.Linear(state_size, 64)
        nn.init.xavier_uniform(self.fcs1.weight)

        self.fc2 = nn.Linear(64 + action_size, 64)
        nn.init.xavier_uniform(self.fc2.weight)

        self.fc3 = nn.Linear(64, 1)
        nn.init.uniform(self.fc3.weight, -EPS, EPS)

    def forward(self, state, action):
        '''
        Forward propogation

        :param state: PyTorch Variable, environment state
        :param action: PyTorch Variable, actions
        :return: PyTorch Tensor a size of 1, which is an evaluation of
                 the policy function estimated by the actor accroding to
                 the Temporal Difference (TD) error
        '''
        s = F.relu(self.fcs1(state))

        x = torch.cat((s, action), dim=1)
        x = F.relu(self.fc2(x))

        out = self.fc3(x)

        return out
