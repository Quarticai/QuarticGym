from torch.utils.data import Dataset
import json
import os
import random
# ---- to capture numpy warnings ----
import warnings

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces, Env  # to create an openai-gym environment https://gym.openai.com/
from mzutils import SimplePriorityQueue, normalize_spaces, denormalize_spaces, mkdir_p
from scipy.integrate import solve_ivp  # the ode solver
from tqdm import tqdm

class TorchDatasetFromD4RL(Dataset):
    def __init__(self, dataset_d4rl) -> None:
        import d3rlpy
        """
        dataset_d4rl should be a dictionary in the d4rl dataset format.
        """
        self.dataset = d3rlpy.dataset.MDPDataset(dataset_d4rl['observations'], dataset_d4rl['actions'],
                                                 dataset_d4rl['rewards'], dataset_d4rl['terminals'])

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        episode = self.dataset.__getitem__(idx)
        return {'observations': episode.observations, 'actions': episode.actions, 'rewards': episode.rewards}


class QuarticGymEnvBase(Env):

    def __init__(self, dense_reward=True, normalize=True, debug_mode=False, action_dim=2, observation_dim=3,
                 reward_function=None, done_calculator=None, max_observations=[1.0, 1.0],
                 min_observations=[-1.0, -1.0], max_actions=[1.0, 1.0], min_actions=[-1.0, -1.0],
                 error_reward=-100.0):
        """the __init__ of a QuarticGym environment.

        Args:
            dense_reward (bool, optional): Whether returns a dense reward or not. If True, will try to return a reward for each step. If False, will return a reward at the end of the episode. Defaults to True.
            normalize (bool, optional): Whether to normalize the actions taken and observations returned to a certain range. If True, the range $$\in R^n, [-1, 1]^n$$ where n is the dimension of an action/observation. Defaults to True.
            debug_mode (bool, optional): Whether to return or print extra information. Defaults to False.
            action_dim (int, optional): Dimensionality of an action. Defaults to 2.
            observation_dim (int, optional): Dimensionality of an observation. Defaults to 3.
            reward_function ([type], optional): Provide to replace with your custom reward function. When not given, use environments' default reward function. Defaults to None.
            done_calculator ([type], optional): Provide to replace with your custom "episode end here calculator". When not given, use environments' default done calculator. Defaults to None.
            max_observations (list, optional): Defaults to [1.0, 1.0].
            min_observations (list, optional): Defaults to [-1.0, -1.0].
            max_actions (list, optional): Defaults to [1.0, 1.0].
            min_actions (list, optional): Defaults to [-1.0, -1.0].
            error_reward (float, optional): When an error is encountered during an episode (this typically means something really bad, like tank overflow). Defaults to -100.0.
        """
        # define arguments
        self.step_count = 0
        self.total_reward = 0
        self.done = False
        self.dense_reward = dense_reward
        self.normalize = normalize  # whether we want to normalize the observation and action to be in between -1 and 1. This is common in most of RL algorithms
        self.debug_mode = debug_mode  # to print debug information.
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.reward_function = reward_function  # if not satisfied with in-house reward function, you can use your own
        self.done_calculator = done_calculator  # if not satisfied with in-house finish calculator, you can use your own
        self.max_observations = max_observations
        self.min_observations = min_observations
        self.max_actions = max_actions
        self.min_actions = min_actions
        self.error_reward = error_reward
        if self.reward_function is None:
            self.reward_function = self.reward_function_standard
        if self.done_calculator is None:
            self.done_calculator = self.done_calculator_standard

        # define the state and action spaces
        self.max_observations = np.array(self.max_observations, dtype=self.np_dtype)
        self.min_observations = np.array(self.min_observations, dtype=self.np_dtype)
        self.max_actions = np.array(self.max_actions, dtype=self.np_dtype)
        self.min_actions = np.array(self.min_actions, dtype=self.np_dtype)
        if self.normalize:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(self.observation_dim,))
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        else:
            self.observation_space = spaces.Box(low=self.min_observations, high=self.max_observations,
                                                shape=(self.observation_dim,))
            self.action_space = spaces.Box(low=self.min_actions, high=self.max_actions, shape=(self.action_dim,))

    def observation_beyond_box(self, observation):
        """
        check if the observation is beyond the box, which is what we don't want.
        """
        return np.any(observation > self.max_observations) or np.any(observation < self.min_observations) or np.any(
            np.isnan(observation)) or np.any(np.isinf(observation))


    def reward_function_standard(self, previous_observation, action, current_observation, reward=None):
        # s, a, r, s, a
        if reward is not None:
            return reward
        elif self.observation_beyond_box(current_observation):
            return self.error_reward

        # TOMODIFY: insert your own reward function here.
        
        reward = max(self.error_reward, reward)  # reward cannot be smaller than the error_reward
        if self.debug_mode:
            print("reward:", reward)
        return reward


    def done_calculator_standard(self, current_observation, step_count, reward, done=None, done_info=None):
        """
        check whether the current episode is considered finished.
        returns a boolean value indicated done or not, and a dictionary with information.
        here in done_calculator_standard, done_info looks like {"terminal": boolean, "timeout": boolean},
        where "timeout" is true when episode end due to reaching the maximum episode length,
        "terminal" is true when "timeout" or episode end due to termination conditions such as env error encountered. (basically done)
        
        """
        if done is None:
            done = False
        else:
            if done_info is not None:
                return done, done_info
            else:
                raise Exception("When done is given, done_info should also be given.")

        if done_info is None:
            done_info = {"terminal": False, "timeout": False}
        else:
            if done_info["terminal"] or done_info["timeout"]:
                done = True
                return done, done_info

        if self.observation_beyond_box(current_observation):
            done_info["terminal"] = True
            done = True
        if reward == self.error_reward:
            done_info["terminal"] = True
            done = True
        if step_count >= self.max_steps:  # same as range(0, max_steps)
            done_info["terminal"] = True
            done_info["timeout"] = True
            done = True

        return done, done_info


    def reset(self, initial_state=None, normalize=None):
        """
        required by gym.
        This function resets the environment and returns an initial observation.
        """
        self.step_count = 0
        self.total_reward = 0
        self.done = False

        if initial_state is not None:
            initial_state = np.array(initial_state, dtype=self.np_dtype)
            observation = initial_state
            self.init_observation = initial_state
        else:
            observation = self.sample_initial_state()
            self.init_observation = observation
        self.previous_observation = observation
        
        # TOMODIFY: reset your environment here.
        
        normalize = self.normalize if normalize is None else normalize
        if normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        return observation


    def step(self, action, normalize=None):
        """
        required by gym.
        This function performs one step within the environment and returns the observation, the reward, whether the episode is finished and debug information, if any.
        """
        if self.debug_mode:
            print("action:", action)
        reward = None
        done = None
        done_info = {"terminal": False, "timeout": False}
        action = np.array(action, dtype=self.np_dtype)
        normalize = self.normalize if normalize is None else normalize
        if normalize:
            action, _, _ = denormalize_spaces(action, self.max_actions, self.min_actions)

        # TOMODIFY: proceed your environment here and collect the observation.
        observation = [0.0, 0.0]

        # compute reward
        if not reward:
            reward = self.reward_function(self.previous_observation, action, observation, reward=reward)
        # compute done
        if not done:
            done, done_info = self.done_calculator(observation, self.step_count, reward, done=done, done_info=done_info)
        self.previous_observation = observation

        self.total_reward += reward
        if self.dense_reward:
            reward = reward  # conventional
        elif not done:
            reward = 0.0
        else:
            reward = self.total_reward
        # clip observation so that it won't be beyond the box
        observation = observation.clip(self.min_observations, self.max_observations)
        if normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        self.step_count += 1
        info = {}
        info.update(done_info)
        return observation, reward, done, info


    def set_initial_states(self, initial_states, num_episodes):
        if initial_states is None:
            initial_states = [self.sample_initial_state() for _ in range(num_episodes)]
        elif isinstance(initial_states, str):
            initial_states = np.load(initial_states)
        assert len(initial_states) == num_episodes
        return initial_states


    def evalute_algorithms(self, algorithms, num_episodes=1, error_reward=-1000.0, initial_states=None, to_plt=True,
                           plot_dir='./plt_results'):
        """
        when excecuting evalute_algorithms, the self.normalize should be False.
        algorithms: list of (algorithm, algorithm_name, normalize). algorithm has to have a method predict(observation) -> action: np.ndarray.
        num_episodes: number of episodes to run
        error_reward: 
        initial_states: None, location of numpy file of initial states or a (numpy) list of initial states
        to_plt: whether generates plot or not
        plot_dir: None or directory to save plots
        returns: list of average_rewards over each episode and num of episodes
        """
        try:
            assert self.normalize is False
        except AssertionError:
            print("env.normalize should be False when executing evalute_algorithms")
            self.normalize = False
        self.error_reward = error_reward
        if plot_dir is not None:
            mkdir_p(plot_dir)
        initial_states = self.set_initial_states(initial_states, num_episodes)
        observations_list = [[] for _ in range(
            len(algorithms))]  # observations_list[i][j][t][k] is algorithm_i_game_j_observation_t_element_k
        actions_list = [[] for _ in
                        range(len(algorithms))]  # actions_list[i][j][t][k] is algorithm_i_game_j_action_t_element_k
        rewards_list = [[] for _ in range(len(algorithms))]  # rewards_list[i][j][t] is algorithm_i_game_j_reward_t
        for n_epi in tqdm(range(num_episodes)):
            for n_algo in range(len(algorithms)):
                algo, algo_name, normalize = algorithms[n_algo]
                algo_observes = []
                algo_actions = []
                algo_rewards = []  # list, for this algorithm, reawards of this trajectory.
                init_obs = self.reset(initial_state=initial_states[n_epi])
                # algo_observes.append(init_obs)
                o = init_obs
                done = False
                while not done:
                    if normalize:
                        o, _, _ = normalize_spaces(o, self.max_observations, self.min_observations)
                    a = algo.predict(o)
                    if normalize:
                        a, _, _ = denormalize_spaces(a, self.max_actions, self.min_actions)
                    algo_actions.append(a)
                    o, r, done, _ = self.step(a)
                    algo_observes.append(o)
                    algo_rewards.append(r)
                observations_list[n_algo].append(algo_observes)
                actions_list[n_algo].append(algo_actions)
                rewards_list[n_algo].append(algo_rewards)

            if to_plt:
                # plot observations
                for n_o in range(self.observation_dim):
                    o_name = self.observation_name[n_o]

                    plt.close("all")
                    plt.figure(0)
                    plt.title(f"{o_name}")
                    for n_algo in range(len(algorithms)):
                        alpha = 1 * (0.7 ** (len(algorithms) - 1 - n_algo))
                        _, algo_name, _ = algorithms[n_algo]
                        plt.plot(np.array(observations_list[n_algo][-1])[:, n_o], label=algo_name, alpha=alpha)
                    plt.plot([initial_states[n_epi][n_o] for _ in range(self.max_steps)], linestyle="--",
                             label=f"initial_{o_name}")
                    plt.xticks(np.arange(1, self.max_steps + 2, 1))
                    plt.annotate(str(initial_states[n_epi][n_o]), xy=(0, initial_states[n_epi][n_o]))
                    plt.legend()
                    if plot_dir is not None:
                        path_name = os.path.join(plot_dir, f"{n_epi}_observation_{o_name}.png")
                        plt.savefig(path_name)
                    plt.close()

                # plot actions
                for n_a in range(self.action_dim):
                    a_name = self.action_name[n_a]

                    plt.close("all")
                    plt.figure(0)
                    plt.title(f"{a_name}")
                    for n_algo in range(len(algorithms)):
                        alpha = 1 * (0.7 ** (len(algorithms) - 1 - n_algo))
                        _, algo_name, _ = algorithms[n_algo]
                        plt.plot(np.array(actions_list[n_algo][-1])[:, n_a], label=algo_name, alpha=alpha)
                    plt.xticks(np.arange(1, self.max_steps + 2, 1))
                    plt.legend()
                    if plot_dir is not None:
                        path_name = os.path.join(plot_dir, f"{n_epi}_action_{a_name}.png")
                        plt.savefig(path_name)
                    plt.close()

                # plot rewards
                plt.close("all")
                plt.figure(0)
                plt.title("reward")
                for n_algo in range(len(algorithms)):
                    alpha = 1 * (0.7 ** (len(algorithms) - 1 - n_algo))
                    _, algo_name, _ = algorithms[n_algo]
                    plt.plot(np.array(rewards_list[n_algo][-1]), label=algo_name, alpha=alpha)
                plt.xticks(np.arange(1, self.max_steps + 2, 1))
                plt.legend()
                if plot_dir is not None:
                    path_name = os.path.join(plot_dir, f"{n_epi}_reward.png")
                    plt.savefig(path_name)
                plt.close()

        observations_list = np.array(observations_list)
        actions_list = np.array(actions_list)
        rewards_list = np.array(rewards_list)
        return observations_list, actions_list, rewards_list


    def generate_dataset_with_algorithm(self, algorithm, normalize=None, num_episodes=1, error_reward=-1000.0,
                                        initial_states=None, format='d4rl'):
        """
        this function aims to create a dataset for offline reinforcement learning, in either d4rl or pytorch format.
        the trajectories are generated by the algorithm, which interacts with this env initialized by initial_states.
        algorithm: an instance that has a method predict(observation) -> action: np.ndarray.
        if format == 'd4rl', returns a dictionary in d4rl format.
        else if format == 'torch', returns an object of type torch.utils.data.Dataset.
        """
        if normalize is None:
            normalize = self.normalize
        initial_states = self.set_initial_states(initial_states, num_episodes)
        dataset = {}
        dataset["observations"] = []
        dataset["actions"] = []
        dataset["rewards"] = []
        dataset["terminals"] = []
        dataset["timeouts"] = []
        for n_epi in tqdm(range(num_episodes)):
            o = self.reset(initial_state=initial_states[n_epi])
            r = 0.0
            done = False
            timeout = False
            final_done = False  # to still record for the last t when done
            while not final_done:
                if done:
                    final_done = True
                # tmp_o is to be normalized, if normalize is true.
                tmp_o = o
                if normalize:
                    tmp_o, _, _ = normalize_spaces(tmp_o, self.max_observations, self.min_observations)
                a = algorithm.predict(tmp_o)
                if normalize:
                    a, _, _ = denormalize_spaces(a, self.max_actions, self.min_actions)
                dataset['observations'].append(o)
                dataset['actions'].append(a)
                dataset['rewards'].append(r)
                dataset['terminals'].append(done)
                dataset["timeouts"].append(timeout)

                o, r, done, info = self.step(a)
                timeout = info['timeout']
        dataset["observations"] = np.array(dataset["observations"])
        dataset["actions"] = np.array(dataset["actions"])
        dataset["rewards"] = np.array(dataset["rewards"])
        dataset["terminals"] = np.array(dataset["terminals"])
        dataset["timeouts"] = np.array(dataset["timeouts"])
        if format == 'd4rl':
            return dataset
        elif format == 'torch':
            return TorchDatasetFromD4RL(dataset)
        else:
            raise ValueError(f"format {format} is not supported.")
