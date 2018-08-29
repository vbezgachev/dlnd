'''
Generic base class for reinforcement learning agents, which provide
capabilities to write statistic into CSV file
'''

import os
import pandas as pd
import torch

from quad_controller_rl import util

class BaseAgent:
    '''
    Generic base class for reinforcement reinforcement agents.
    Also writes statistic and saves the model onto the the disk
    '''

    def __init__(self, task):
        '''Initialize policy and other agent parameters.

        :param task: Should be able to access the following (OpenAI Gym spaces):
            task.observation_space  # i.e. state space
            task.action_space
        '''
        # init statistics writing
        self.stats_dir = util.get_param('out')
        if not os.path.exists(self.stats_dir):
            os.makedirs(self.stats_dir)

        self.stats_filename = os.path.join(
            self.stats_dir,
            'stats_{}.csv'.format(util.get_timestamp())
        )
        self.stats_columns = ['episode', 'total_reward']
        print('Saving statistics {} to {}'.format(self.stats_columns, self.stats_filename))

        # init models writing
        self.models_dir = util.get_param('models')
        self.actor_best_model_file = self.models_dir + '/actor_best.pth'
        self.critic_best_model_file = self.models_dir + '/critic_best.pth'
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def step(self, state, reward, done):
        '''Process state, reward, done flag, and return an action.

        :param state: current state vector as Numpy array, compatible with task's state space
        :param reward: last reward received
        :param done: whether this episode is complete

        :return: desired action vector as NumPy array, compatible with task's action space
        '''
        raise NotImplementedError("{} must override step()".format(self.__class__.__name__))

    def write_episode_stats(self, episode_num, total_reward):
        '''
        Write single episode total reward into CSV

        :param episode_num: Episode numbers
        :param total_reward: Total reward received
        '''
        df_stats = pd.DataFrame([[episode_num, total_reward]], columns=self.stats_columns)
        df_stats.to_csv(
            self.stats_filename,
            mode='a', # append to the end of file
            index=False,
            header=not os.path.isfile(self.stats_filename) # write headers first time only
            )

    def save_models(self, episode_num, actor_model, critic_model, is_best):
        '''
        Saves models permanently

        :param episode_num: Episode numbers
        :param actor_model: (Target) actor model to save
        :param critic_model: (Target) critic model to save
        :param is_best: Did the models achieves best results
        '''
        if is_best:
            torch.save(actor_model.state_dict(), self.actor_best_model_file)
            torch.save(critic_model.state_dict(), self.critic_best_model_file)

        torch.save(actor_model.state_dict(),
                   self.models_dir + '/' + str(episode_num) + '_actor.pth')
        torch.save(critic_model.state_dict(),
                   self.models_dir + '/' + str(episode_num) + '_critic.pth')

    def load_models(self, actor_model, critic_model):
        '''
        Loads best actor and critic models if found

        :param actor_model: Actor model to load
        :param critic_model: Critic model to load

        :return: True if the best actor and critic models have been found and loaded,
                 False - otherwise
        '''
        if os.path.exists(self.actor_best_model_file) and\
           os.path.exists(self.critic_best_model_file):

            actor_model.load_state_dict(torch.load(self.actor_best_model_file))
            critic_model.load_state_dict(torch.load(self.critic_best_model_file))
            return True

        return False
