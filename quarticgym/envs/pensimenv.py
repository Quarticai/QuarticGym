'''
import math
import csv
import codecs
import sys
import random
import numpy as np
from .utils import *
from gym import spaces, Env
from pensimpy.peni_env_setup import PenSimEnv


csv.field_size_limit(sys.maxsize)
random.seed(0)
MINUTES_PER_HOUR = 60
BATCH_LENGTH_IN_MINUTES = 230 * MINUTES_PER_HOUR
BATCH_LENGTH_IN_HOURS = 230
STEP_IN_MINUTES = 12
STEP_IN_HOURS = STEP_IN_MINUTES / MINUTES_PER_HOUR
NUM_STEPS = int(BATCH_LENGTH_IN_MINUTES / STEP_IN_MINUTES)
WAVENUMBER_LENGTH = 2200


def get_observation_data_reformed(observation, t):
    """
    Get observation data at t.
    vars are Temperature,Acid flow rate,Base flow rate,Cooling water,Heating water,Vessel Weight,Dissolved oxygen concentration 
    respectively in csv terms, but used abbreviation here to stay consistent with peni_env_setup
    """
    vars = ['T', 'Fa', 'Fb', 'Fc', 'Fh', 'Wt', 'DO2']
    pH = observation.pH.y[t]
    pH = -math.log(pH) / math.log(10) if pH != 0 else pH
    return [t * STEP_IN_MINUTES / MINUTES_PER_HOUR, pH] + [eval(f"observation.{var}.y[t]", {'observation': observation, 't': t}) for var in vars]



class PenSimEnvGym(PenSimEnv, Env):
    def __init__(self, recipe_combo, dense_reward=True, observation_dim=9, action_dim=6, normalize=True, observation_relaxation=1.2, fast=True):
        """
        Time is not in our observation_space. We make the env time unaware and MDP.
        :dense_reward : bool, if True we return reward for each step, if False we accumulate the reward and return the sum at the end of an episode
        :observation_dim : int the dimensionality of state
        :action_dim : int the dimensionality of action
        :normalize : bool, whether to normalize the observation and action speace
        :observation_relaxation : we need observation_relaxation since the max_observation and the min_observation are 
        set by the loaded dataset, and there might be more extreme cases encountered in the future exploration. 
        we set the default to be 1.2 so there are 20% room if you are confident that no future observations 
        will go beyond your current max/min observation, you may want to set observation_relaxation=1
        """
        super(PenSimEnvGym, self).__init__(recipe_combo, fast=fast)
        self.dense_reward = dense_reward
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.observation_space = spaces.Box(low=-1*observation_relaxation, high=1*observation_relaxation, shape=(self.observation_dim,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        self._max_episode_steps = NUM_STEPS
        # ---- set by dataset or use predefined as you wish if applicable ----
        self.normalize = normalize
        self.max_observations = 1
        self.min_observations = -1
        self.max_actions = np.array([4100.0, 151.0, 36.0, 76.0, 1.2, 510.0])
        self.min_actions = np.array([0.0, 7.0, 21.0, 29.0, 0.5, 0.0])
        # there are 6 items: Discharge rate,Sugar feed rate,Soil bean feed rate,Aeration rate,Back pressure,Water injection/dilution
        # ---- set by dataset or use predefined as you wish if applicable ----

    def reset(self):
        _, x = super().reset()
        self.x = x
        self.k = 0
        self.total_reward = 0
        observation = get_observation_data_reformed(x, 0)
        observation = np.array(observation, dtype=np.float32)
        if self.normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)

        return observation

    def step(self, action):
        """
        actions in action (list) are in the order [discharge, Fs, Foil,Fg, pressure, Fw]
        """
        action = np.array(action, dtype=np.float32)
        if self.normalize:
            action, _, _ = denormalize_spaces(action, self.max_actions, self.min_actions)
        self.k += 1 
        values_dict = self.recipe_combo.get_values_dict_at(self.k * STEP_IN_MINUTES)
        # served as a batch buffer below
        pensimpy_observation, x, yield_per_run, done = super().step(self.k, self.x, action[1], action[2], action[3], action[4], action[0], action[5], values_dict['Fpaa'])
        self.x = x
        new_observation = get_observation_data_reformed(x, self.k - 1)
        new_observation = np.array(new_observation, dtype=np.float32)
        if self.normalize:
            new_observation, _, _ = normalize_spaces(new_observation, self.max_observations, self.min_observations)

        # dense or sparse reward
        self.total_reward += yield_per_run
        if self.dense_reward:
            reward = yield_per_run
        elif not done:
            reward = 0.0
        else:
            reward = self.total_reward
        return new_observation, reward, done, {}
        # state, reward, done, info in gym env term


class PeniControlData:
    """
    dataset class helper, mainly aims to mimic d4rl's qlearning_dataset format (which returns a dictionary).
    produced from PenSimPy generated csvs.
    """
    def __init__(self, load_just_a_file='', dataset_folder='examples/example_batches', delimiter=',', observation_dim=9, action_dim=6) -> None:
        """
        :param dataset_folder: where all dataset csv files are living in
        """
        self.dataset_folder = dataset_folder
        self.delimiter = delimiter
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        if load_just_a_file!= '':
            file_list = [load_just_a_file]
        else:
            file_list = get_things_in_loc(dataset_folder, just_files=True)
        self.file_list = file_list

    def load_file_list_to_dict(self, file_list, shuffle=True):
        file_list = file_list.copy()
        random.shuffle(file_list)
        dataset= {}
        observations = []
        actions = []
        next_observations = []
        rewards = []
        terminals = []
        for file_path in file_list:
            tmp_observations = []
            tmp_actions = []
            tmp_next_observations = []
            tmp_rewards = []
            tmp_terminals = []
            with codecs.open(file_path, 'r', encoding='utf-8') as fp:
                csv_reader = csv.reader(fp, delimiter=self.delimiter)
                next(csv_reader) 
                # get rid of the first line containing only titles
                for row in csv_reader:
                    observation = [row[0]] + row[7:-1] 
                    # there are 9 items: Time Step, pH,Temperature,Acid flow rate,Base flow rate,Cooling water,Heating water,Vessel Weight,Dissolved oxygen concentration
                    assert len(observation) == self.observation_dim
                    action = [row[1], row[2], row[3], row[4], row[5], row[6]] 
                    # there are 6 items: Discharge rate,Sugar feed rate,Soil bean feed rate,Aeration rate,Back pressure,Water injection/dilution
                    assert len(action) == self.action_dim
                    reward = row[-1]
                    terminal = False
                    tmp_observations.append(observation)
                    tmp_actions.append(action)
                    tmp_rewards.append(reward)
                    tmp_terminals.append(terminal)
            tmp_terminals[-1] = True
            tmp_next_observations = tmp_observations[1:] + [tmp_observations[-1]]
            observations += tmp_observations
            actions += tmp_actions
            next_observations +=tmp_next_observations
            rewards += tmp_rewards
            terminals += tmp_terminals
        dataset['observations'] = np.array(observations, dtype=np.float32)
        dataset['actions'] = np.array(actions, dtype=np.float32)
        dataset['next_observations'] = np.array(next_observations, dtype=np.float32)
        dataset['rewards'] = np.array(rewards, dtype=np.float32)
        dataset['terminals'] = np.array(terminals, dtype=bool)
        self.max_observations = dataset['observations'].max(axis=0)
        self.min_observations = dataset['observations'].min(axis=0)
        dataset['observations'], _, _ = normalize_spaces(dataset['observations'], self.max_observations, self.min_observations)
        dataset['next_observations'], _, _ = normalize_spaces(dataset['next_observations'], self.max_observations, self.min_observations)
        self.max_actions = dataset['actions'].max(axis=0)
        self.min_actions = dataset['actions'].min(axis=0)
        dataset['actions'], _, _ = normalize_spaces(dataset['actions'], self.max_actions, self.min_actions) # passed in a normalized version.
        # self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        return dataset

    def get_dataset(self):
        return self.load_file_list_to_dict(self.file_list)
'''
