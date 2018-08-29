'''Policy search agent - a Deep Determenistic Policy Gradients'''

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from quad_controller_rl.agents.base_agent import BaseAgent
from quad_controller_rl.agents.model import Actor, Critic
from quad_controller_rl.agents.ou_noise import OrnsteinUhlenbeckProcess
from quad_controller_rl.agents.replay_buffer import ReplayBuffer

class DDPG(BaseAgent):
    '''Agent that searches for optimal policy using Deep Deterministic Policy Gradients.'''

    def __init__(self, task):
        '''
        Initializes variables

        :param task: Should be able to access the following (OpenAI Gym spaces):
            task.observation_space  # i.e. state space
            task.action_space
        '''
        super(DDPG, self).__init__(task)

        self.use_gpu = torch.cuda.is_available()
        self.task = task

        # Hyperparameters
        self.gamma = 0.99 # discount factor
        self.tau = 0.001 # for sort update of target parameters

        # constrained states
        self.state_size = np.prod(self.task.observation_space.shape).item()

        # constrained actions
        self.action_size = 1
        self.action_low = self.task.action_space.low[2:3]
        self.action_high = self.task.action_space.high[2:3]

        # Actor model
        self.actor_local = Actor(self.state_size, self.action_size,
                                 self.action_low, self.action_high,
                                 self.use_gpu)
        self.actor_target = Actor(self.state_size, self.action_size,
                                  self.action_low, self.action_high,
                                  self.use_gpu)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), 1e-4)

        # Critic model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), 1e-3)

        # load the models and sync weights target models
        self.best_model_loaded = self.load_models(self.actor_local, self.critic_local)
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        print('Best model loaded: {}'.format(self.best_model_loaded))

        # use GPU?
        if self.use_gpu:
            self.actor_local.cuda()
            self.actor_target.cuda()
            self.critic_local.cuda()
            self.critic_target.cuda()

        # Ornstein-Uhlenbeck noise for action sampling
        self.noise = OrnsteinUhlenbeckProcess(
            size=self.action_size, theta=0.15, sigma=0.02)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 128
        self.memory = ReplayBuffer(self.buffer_size)

        # Score tracker and learning parameters
        self.best_score = -np.inf

        # Episode variables
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        self.episode_num = 1
        self.acts = np.zeros(shape=self.task.action_space.shape) # actions to reuturn from step()
                                                                 # we set all actions to 0
                                                                 # except one for vertical forces

    def reset_episode_vars(self):
        '''Resets episode variables'''
        self.last_state = None
        self.last_action = None
        self.total_reward = 0.0
        self.count = 0
        self.episode_num += 1
        self.acts = np.zeros(shape=self.task.action_space.shape)

    def step(self, state, reward, done):
        '''Process state, reward, done flag, and return an action.

        :param state: current state vector as Numpy array, compatible with task's state space
        :param reward: last reward received
        :param done: whether this episode is complete

        :return: desired action vector as NumPy array, compatible with task's action space
        '''
        # Choose an action
        state = state[0:self.state_size]
        action = self.act(state)

        # Save experience / reward
        if self.last_state is not None and self.last_action is not None:
            self.memory.add(self.last_state, self.last_action, reward, state, done)
            self.total_reward += reward
            self.count += 1

        # Learn, if we have enough samples
        if len(self.memory) > self.batch_size and not self.best_model_loaded:
            experience = self.memory.sample(self.batch_size)
            self.learn(experience)

        # Write statistic and saves model if done
        if done:
            score = self.total_reward / float(self.count) if self.count else 0.0
            if score > self.best_score:
                self.best_score = score
                self.save_models(self.episode_num, self.actor_target, self.critic_target, True)

            print("DDPG.learn(): t = {:4d}, score = {:7.3f} (best = {:7.3f}), total reward = {:7.3f}, episode = {}".format(
                  self.count, score, self.best_score, self.total_reward, self.episode_num))
            
            if self.episode_num % 10 == 0:
                self.save_models(self.episode_num, self.actor_target, self.critic_target, False)

            self.write_episode_stats(self.episode_num, self.total_reward)
            self.reset_episode_vars()

        # saves last state and actions
        self.last_state = state
        self.last_action = action
        self.acts[2] = action # change only vertical forces

        return self.acts

    def act(self, state):
        '''
        Predict actions for a state

        :param state: Numpy array, environment state
        :return: Numpy array, predicted actions
        '''
        state = self.to_var(torch.from_numpy(state).float())

        self.actor_local.eval()
        action = self.actor_local.forward(state).detach()
        return action.data.cpu().numpy() + self.noise.sample()

    def to_var(self, x_numpy):
        '''
        Helper to convert Numpy array to PyTorch tensor

        :param x_numpy: Numpy array to convert
        :return: PyTorch tensor
        '''
        x_var = Variable(x_numpy)
        if self.use_gpu:
            x_var = x_var.cuda()
        return x_var

    def learn(self, experiences):
        '''
        Trains the networks

        :param experiences: tuple of the experience - (states, actions, rewards, next_states, dones)
        '''
        # -------------------- get data from batch --------------------
        # get expereiences from the replay buffer
        states = np.vstack(experiences[0])
        states = self.to_var(torch.from_numpy(states).float())

        actions = np.vstack(experiences[1])
        actions = self.to_var(torch.from_numpy(actions).float())

        rewards = np.float32(experiences[2])
        rewards = self.to_var(torch.from_numpy(rewards))
        rewards = torch.unsqueeze(rewards, 1)

        next_states = np.vstack(experiences[3])
        next_states = self.to_var(torch.from_numpy(next_states).float())

        dones = np.float32(experiences[4])
        not_dones = self.to_var(torch.from_numpy(1 - dones))
        not_dones = torch.unsqueeze(not_dones, 1)

        # ---------------------- optimize critic ----------------------
        next_actions = self.actor_target.forward(next_states).detach()
        Q_targets_next = self.critic_target.forward(next_states, next_actions).detach()

        Q_targets_next = not_dones * Q_targets_next
        Q_targets = rewards + (self.gamma * Q_targets_next)

        Q_predicted = self.critic_local.forward(states, actions)

        # compute critic model loss and train it
        value_loss = nn.SmoothL1Loss()(Q_predicted, Q_targets)

        self.critic_local.zero_grad()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # ---------------------- optimize actor -----------------------
        predicted_actions = self.actor_local.forward(states)
        policy_loss = torch.mean(-self.critic_local.forward(states, predicted_actions))

        self.actor_local.zero_grad()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # soft update of target models
        self.soft_update(self.actor_target, self.actor_local)
        self.soft_update(self.critic_target, self.critic_local)

    def hard_update(self, target_model, local_model):
        '''
        Hard update of the target model weights - just copy them from the local model

        :param target_model: Destination, target model
        :param local_model: Source, local model
        '''
        for target_param, param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target_model, local_model):
        '''
        Soft update of the target model weights corresponding to DDPG algorithm

        :param target_model: Destination, target model
        :param local_model: Source, local model
        '''
        for target_param, param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
