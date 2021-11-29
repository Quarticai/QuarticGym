from scipy.integrate import solve_ivp # the ode solver
import numpy as np
from gym import spaces, Env # to create an openai-gym environment https://gym.openai.com/
from tqdm import tqdm
from .utils import *
import json

import matplotlib.pyplot as plt
import os
# ---- to capture numpy warnings ---- 
import warnings
np.seterr(all='warn')
# ---- to capture numpy warnings ---- 


# macros the defined by the reactor
MAX_OBSERVATIONS = [1.0, 100.0, 1.0] # cA, T, h
MIN_OBSERVATIONS = [1e-08, 1e-08, 1e-08]
MAX_ACTIONS = [35.0, 0.2] # Tc, qout
MIN_ACTIONS = [15.0, 0.05]
STEADY_OBSERVATIONS = [0.8778252, 51.34660837, 0.659]
STEADY_ACTIONS = [26.85, 0.1]
ERROR_REWARD = -1000.0

class ReactorModel:
    
    def __init__(self, sampling_time):
        
        # Define reactor model parameters
        self.q_in = .1        # m^3/min
        self.Tf = 76.85     # degrees C
        self.cAf = 1.0       # kmol/m^3
        self.r = .219       # m
        self.k0 = 7.2e10    # min^-1
        self.E_divided_by_R = 8750       # K
        self.U = 54.94      # kg/min/m^2/K
        self.rho = 1000     # kg/m^3
        self.Cp = .239      # kJ/kg/K
        self.dH = -5e4      # kJ/kmol
        
        self.Nx = 3         # Number of state variables
        self.Nu = 2         # Number of input variables
        
        self.sampling_time = sampling_time      # sampling time or integration step
    
    # Ordinary Differential Equations (ODEs) described in the report i.e. Equations (1), (2), (3)
    def ode(self, x, u):
    
        c = x[0]        # c_A
        T = x[1]        # T
        h = x[2]        # h
        Tc = u[0]       # Tc
        q = u[1]        # q_out
        
        rate = self.k0*c*np.exp(-self.E_divided_by_R / (T + 273.15))  # kmol/m^3/min

        dxdt = [
            self.q_in * (self.cAf - c) / (np.pi * self.r ** 2 * h + 1e-8) - rate,  # kmol/m^3/min
            self.q_in * (self.Tf - T) / (np.pi * self.r ** 2 * h + 1e-8)
            - self.dH / (self.rho * self.Cp) * rate
            + self.U / (np.pi * self.r ** 2 * h * self.rho * self.Cp + + 1e-8) * (Tc - T),  # degree C/min
            (self.q_in - q) / (np.pi * self.r ** 2)  # m/min
        ]
        return dxdt
    
    
    # integrates one sampling time or time step and returns the next state
    def step(self, x, u):
        # return self.simulator.sim(x,u)
        sol = solve_ivp(lambda t,x,u:self.ode(x,u), [0, self.sampling_time], x, args=(u,), method='LSODA')
        return sol.y[:,-1]


class ReactorEnv(Env):

    def __init__(self, dense_reward=True, normalize=True, debug_mode=False, action_dim=2, observation_dim=3, reward_function=None, done_calculator=None, max_observations=MAX_OBSERVATIONS, min_observations=MIN_OBSERVATIONS, max_actions=MAX_ACTIONS, min_actions=MIN_ACTIONS, error_reward=ERROR_REWARD, # general env inputs
    initial_state_deviation_ratio=0.3, compute_diffs_on_reward=False, np_dtype=np.float32, sampling_time=0.1, max_steps=100):
        # ---- standard ----
        # define arguments
        self.step_count = 0
        self.total_reward = 0
        self.done = False
        self.dense_reward = dense_reward
        self.normalize = normalize # whether we want to normalize the observation and action to be in between -1 and 1. This is common in most of RL algorithms
        self.debug_mode = debug_mode # to print debug information.
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.reward_function = reward_function # if not satisfied with in-house reward function, you can use your own
        self.done_calculator = done_calculator # if not satisfied with in-house finish calculator, you can use your own
        self.max_observations = max_observations
        self.min_observations = min_observations
        self.max_actions = max_actions
        self.min_actions = min_actions
        self.error_reward = error_reward
        if self.reward_function is None:
            self.reward_function = self.reward_function_standard
        if self.done_calculator is None:
            self.done_calculator = self.done_calculator_standard
        # /---- standard ----

        self.compute_diffs_on_reward = compute_diffs_on_reward # how the reward is computed, if True, then the reward is computed as the difference between the current state and the previous state
        self.np_dtype = np_dtype
        self.sampling_time = sampling_time
        self.max_steps = max_steps

        self.observation_name = ["cA", "T", "h"]
        self.action_name = ["Tc", "qout"]

        # ---- standard ----
        # define the state and action spaces
        self.max_observations = np.array(self.max_observations, dtype=self.np_dtype)
        self.min_observations = np.array(self.min_observations, dtype=self.np_dtype)
        self.max_actions = np.array(self.max_actions, dtype=self.np_dtype)
        self.min_actions = np.array(self.min_actions, dtype=self.np_dtype)
        if self.normalize:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(self.observation_dim,))
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        else:
            self.observation_space = spaces.Box(low=self.min_observations, high=self.max_observations, shape=(self.observation_dim,))
            self.action_space = spaces.Box(low=self.min_actions, high=self.max_actions, shape=(self.action_dim,))
        # /---- standard ----
        
        self.steady_observations = np.array(STEADY_OBSERVATIONS, dtype=self.np_dtype) # cA, T, h
        self.steady_actions = np.array(STEADY_ACTIONS, dtype=self.np_dtype) # Tc, qout
        self.initial_state_deviation_ratio = initial_state_deviation_ratio

    # ---- standard ----
    def observation_beyond_box(self, observation):
        """
        check if the observation is beyond the box, which is what we don't want.
        """
        return np.any(observation > self.max_observations) or np.any(observation < self.min_observations) or np.any(np.isnan(observation)) or np.any(np.isinf(observation))
    # /---- standard ----

    def reward_function_standard(self, previous_observation, action, current_observation, reward=None):
        # ---- standard ----
        # s, a, r, s, a
        if reward is not None:
            return reward
        elif self.observation_beyond_box(current_observation):
            return self.error_reward
        # /---- standard ----
        current_observation_evaluated = self.evaluate_observation(current_observation)
        assert isinstance(current_observation_evaluated, float)
        if self.compute_diffs_on_reward:
            previous_observation_evaluated = self.evaluate_observation(previous_observation)
            assert isinstance(previous_observation_evaluated, float)
            reward = current_observation_evaluated - previous_observation_evaluated
        else:
            reward = current_observation_evaluated
        # ---- standard ----
        reward = max(self.error_reward, reward) # reward cannot be smaller than the error_reward
        if self.debug_mode:
            print("reward:", reward)
        return reward
        # /---- standard ----
    
    # ---- standard ----
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
        if step_count >= self.max_steps: # same as range(0, max_steps)
            done_info["terminal"] = True
            done_info["timeout"] = True
            done = True
        
        return done, done_info
    # /---- standard ----

    def reset(self, initial_state=None, normalize=None):
        # ---- standard ----
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
        # /---- standard ----
        self.reactor = ReactorModel(self.sampling_time)

        # ---- standard ----
        normalize = self.normalize if normalize is None else normalize
        if normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        return observation
        # /---- standard ----

    def step(self, action, normalize=None):
        # ---- standard ----
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
        # /---- standard ----

        # ---- to capture numpy warnings ---- 
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("error")
            try:
                observation = self.reactor.step(self.previous_observation, action)
            except Exception as e:
                print("Got Exception/Warning: ", e)
                observation = self.previous_observation
                reward = self.error_reward
                done = True
                done_info["terminal"] = True
        # /---- to capture numpy warnings ---- 

        # ---- standard ----
        # compute reward
        if not reward:
            reward = self.reward_function(self.previous_observation, action, observation, reward=reward)
        # compute done
        if not done:
            done, done_info = self.done_calculator(observation, self.step_count, reward, done=done, done_info=done_info)
        self.previous_observation = observation

        self.total_reward += reward
        if self.dense_reward:
            reward = reward # conventional
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
        # /---- standard ----
        
    def evenly_spread_initial_states(self, val_per_state, dump_location=None):
        initial_state_deviation_ratio=self.initial_state_deviation_ratio
        steady_observations=self.steady_observations
        len_obs = len(steady_observations)
        val_range = val_per_state**len_obs
        initial_states = np.zeros([val_range,len_obs])
        tmp_o = []
        for oi in range(len_obs):
            tmp_o.append(np.linspace(steady_observations[oi] * (1.0-initial_state_deviation_ratio), steady_observations[oi] * (1.0+initial_state_deviation_ratio), num=val_per_state, endpoint=True))

        for i in range(val_range):
            tmp_val_range = i
            curr_val = []
            for oi in range(len_obs):
                rmder = tmp_val_range % val_per_state
                curr_val.append(tmp_o[oi][rmder])
                tmp_val_range = int((tmp_val_range - rmder) / val_per_state)
            initial_states[i] = curr_val
        if dump_location is not None:
            np.save(dump_location, initial_states)
        return initial_states
    
    # ---- standard ----
    def set_initial_states(self, initial_states, num_episodes):
        if initial_states is None:
            initial_states = [self.sample_initial_state() for _ in range(num_episodes)]
        elif isinstance(initial_states, str):
            initial_states = np.load(initial_states)
        assert len(initial_states) == num_episodes
        return initial_states
    # /---- standard ----

    def evalute_algorithms(self, algorithms, num_episodes=1, error_reward=-1000.0, initial_states=None, plot_dir='./plt_results'):
        # ---- standard ----
        """
        when excecuting evalute_algorithms, the self.normalize should be False.
        algorithms: list of (algorithm, algorithm_name, normalize). algorithm has to have a method predict(observation) -> action: np.ndarray.
        num_episodes: number of episodes to run
        error_reward: to work with Xiaozhou's evaluation script
        initial_states: None, location of numpy file of initial states or a (numpy) list of initial states
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
        observations_list = [[] for _ in range(len(algorithms))] # observations_list[i][j][t][k] is algorithm_i_game_j_observation_t_element_k
        actions_list = [[] for _ in range(len(algorithms))] # actions_list[i][j][t][k] is algorithm_i_game_j_action_t_element_k
        rewards_list = [[] for _ in range(len(algorithms))] # rewards_list[i][j][t] is algorithm_i_game_j_reward_t
        for n_epi in tqdm(range(num_episodes)):
            for n_algo in range(len(algorithms)):
                algo, algo_name, normalize = algorithms[n_algo]
                algo_observes = []
                algo_actions = []
                algo_rewards = [] #list, for this algorithm, reawards of this trajectory.
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
            # plot observations
            for n_o in range(self.observation_dim):
                o_name = self.observation_name[n_o]

                plt.close("all")
                plt.figure(0)
                plt.title(f"{o_name}")
                for n_algo in range(len(algorithms)):
                    _, algo_name, _ = algorithms[n_algo]
                    plt.plot(np.array(observations_list[n_algo][-1])[:, n_o], label=algo_name)
                plt.plot([initial_states[n_epi][n_o] for _ in range(self.max_steps)], linestyle="--", label=f"initial_{o_name}")
                plt.plot([self.steady_observations[n_o] for _ in range(self.max_steps)], linestyle="-.", label=f"steady_{o_name}")
                plt.xticks(np.arange(1, self.max_steps + 2, 1))
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
                    _, algo_name, _ = algorithms[n_algo]
                    plt.plot(np.array(actions_list[n_algo][-1])[:, n_a], label=algo_name)
                plt.plot([self.steady_actions[n_a] for _ in range(self.max_steps)], linestyle="-.", label=f"steady_{a_name}")
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
                _, algo_name, _ = algorithms[n_algo]
                plt.plot(np.array(rewards_list[n_algo][-1]), label=algo_name)
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
        # /---- standard ----
        
    def evaluate_rewards_mean_std_over_episodes(self,algorithms, num_episodes=1, error_reward=-1000.0, initial_states=None, plot_dir='./plt_results'):
        """
        returns: mean and std of rewards over all episodes
        """
        result_dict = {}
        observations_list, actions_list, rewards_list = self.evalute_algorithms(algorithms, num_episodes=num_episodes, error_reward=error_reward, initial_states=initial_states, plot_dir=plot_dir)
        for n_algo in range(len(algorithms)):
            _, algo_name, _ = algorithms[n_algo]
            rewards_list_curr_algo = rewards_list[n_algo]
            rewards_mean_over_episodes = [] # rewards_mean_over_episodes[n_epi] is mean of rewards of n_epi
            for n_epi in range(num_episodes):
                if rewards_list_curr_algo[n_epi][-1] == error_reward:
                    rewards_mean_over_episodes.append(error_reward)
                else:
                    rewards_mean_over_episodes.append(np.mean(rewards_list_curr_algo[n_epi]))
            rewards_mean = np.mean(rewards_mean_over_episodes)
            rewards_std = np.std(rewards_mean_over_episodes)
            print(f"{algo_name}_reward_mean: {rewards_mean}")
            result_dict[algo_name + "_reward_mean"] = rewards_mean
            print(f"{algo_name}_reward_std: {rewards_std}")
            result_dict[algo_name + "_reward_std"] = rewards_std
        json.dump(result_dict, open(os.path.join(plot_dir, 'result.json'), 'w+'))
        return observations_list, actions_list, rewards_list

    def sample_initial_state(self):
        init_observation = np.maximum(np.random.uniform(low=(1-self.initial_state_deviation_ratio)*self.steady_observations, high=(1+self.initial_state_deviation_ratio)*self.steady_observations), 0, 
            dtype=self.np_dtype)
        init_observation = init_observation.clip(self.min_observations, self.max_observations)
        return init_observation

    # ---- standard ----
    def evaluate_observation(self, observation):
        """
        observation: numpy array of shape (self.observation_dim)
        returns: observation evaluation (reward in a sense)
        """
        
        return float( - ( np.mean( (observation - self.steady_observations) ** 2 / np.maximum((self.init_observation - self.steady_observations) ** 2, 1e-8) ) ) )
    # /---- standard ----

    # ---- standard ----
    def generate_dataset_with_algorithm(self, algorithm, normalize=None, num_episodes=1, error_reward=-1000.0, initial_states=None, format='d4rl'):
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
            final_done = False # to still record for the last t when done
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
    # /---- standard ----
