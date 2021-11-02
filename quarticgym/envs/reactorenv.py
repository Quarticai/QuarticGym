from scipy.integrate import solve_ivp # the ode solver
import numpy as np
from gym import spaces, Env # to create an openai-gym environment https://gym.openai.com/
from tqdm import tqdm
from .utils import *

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
ERROR_REWARD = -100.0

class ReactorModel:
    
    def __init__(self, sampling_time):
        
        # Define reactor model parameters
        self.q_in = .1        # m^3/min
        self.Tf = 76.85     # degrees C
        self.cAf = 1.0       # kmol/m^3
        self.r = .219       # m
        self.k0 = 7.2e10    # min^-1
        self.E = 8750       # K
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
        
        rate = self.k0*c*np.exp(-self.E/(T+273.15))  # kmol/m^3/min
        
        dxdt = [
            self.q_in*(self.cAf - c)/(np.pi*self.r**2*h + 1e-8) - rate, # kmol/m^3/min
            self.q_in*(self.Tf - T)/(np.pi*self.r**2*h + 1e-8) 
                        - self.dH/(self.rho*self.Cp + 1e-8)*rate
                        + 2*self.U/(self.r*self.rho*self.Cp + 1e-8)*(Tc - T), # degree C/min
            (self.q_in - q)/(np.pi*self.r**2 + 1e-8)     # m/min
                ]
        return dxdt
    
    
    # integrates one sampling time or time step and returns the next state
    def step(self, x, u):
        # return self.simulator.sim(x,u)
        sol = solve_ivp(lambda t,x,u:self.ode(x,u), [0, self.sampling_time], x, args=(u,), method='LSODA')
        return sol.y[:,-1]


class ReactorEnv(Env):

    def __init__(self, dense_reward=True, normalize=True, debug_mode=False, action_dim=2, observation_dim=3, reward_function=None, done_calculator=None, max_observations=MAX_OBSERVATIONS, min_observations=MIN_OBSERVATIONS, max_actions=MAX_ACTIONS, min_actions=MIN_ACTIONS, error_reward=ERROR_REWARD, # general env inputs
    initial_state_scale=[0.25, 25, 0.2], compute_diffs_on_reward=False, np_dtype=np.float32, sampling_time=0.1, max_steps=100):
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

        self.initial_state_scale = initial_state_scale # to use by the initial_state_generator
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

    # ---- standard ----
    def observation_beyond_box(self, observation):
        """
        check if the observation is beyond the box, which is what we don't want.
        """
        return np.any(observation > self.max_observations) or np.any(observation < self.min_observations)
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
    def done_calculator_standard(self, current_observation, step_count, done=None):
        """
        check whether the current episode is considered finished.
        """
        if done is not None:
            return done
        elif self.observation_beyond_box(current_observation):
            return True
        if step_count >= self.max_steps: # same as range(0, max_steps)
            return True
        else:
            return False
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
        # /---- to capture numpy warnings ---- 

        # ---- standard ----
        # compute reward
        reward = self.reward_function(self.previous_observation, action, observation, reward=reward)
        # compute done
        done = self.done_calculator(observation, self.step_count, done=done)
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
        return observation, reward, done, {}
        # /---- standard ----

    def evalute_algorithms(self, algorithms, num_episodes=1, initial_states=None, plot_dir='./plt_results'):
        # ---- standard ----
        """
        when excecuting evalute_algorithms, the self.normalize should be False.
        algorithms: list of (algorithm, algorithm_name, normalize). algorithm has to have a method predict(observation) -> action: np.ndarray.
        num_episodes: number of episodes to run
        initial_states: None or list of initial states
        plot_dir: None or directory to save plots
        returns: list of average_rewards over each episode and num of episodes
        """
        try:
            assert self.normalize is False
        except AssertionError:
            print("env.normalize should be False when executing evalute_algorithms")
            self.normalize = False
        if plot_dir is not None:
            mkdir_p(plot_dir)
        if initial_states is None:
            initial_states = [self.sample_initial_state() for _ in range(num_episodes)]
        else:
            assert len(initial_states) == num_episodes
        mean_rewards = [[] for _ in range(len(algorithms))]
        for n_epi in tqdm(range(num_episodes)):
            total_observes = []
            total_actions = []
            total_rewards = []
            for n_algo in range(len(algorithms)):
                algo, algo_name, normalize = algorithms[n_algo]
                algo_observes = []
                algo_actions = []
                algo_rewards = []
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
                total_observes.append(np.array(algo_observes))
                total_actions.append(np.array(algo_actions))
                total_rewards.append(np.array(algo_rewards))
                mean_rewards[n_algo].append(np.mean(algo_rewards))
            # plot observations
            for n_o in range(self.observation_dim):
                o_name = self.observation_name[n_o]

                plt.close("all")
                plt.figure(0)
                plt.title(f"{o_name}")
                for n_algo in range(len(algorithms)):
                    _, algo_name, _ = algorithms[n_algo]
                    plt.plot(total_observes[n_algo][:, n_o], label=algo_name)
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
                    plt.plot(total_actions[n_algo][:, n_a], label=algo_name)
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
                plt.plot(total_rewards[n_algo], label=algo_name)
            plt.xticks(np.arange(1, self.max_steps + 2, 1))
            plt.legend()
            if plot_dir is not None:
                path_name = os.path.join(plot_dir, f"{n_epi}_reward.png")
                plt.savefig(path_name)
            plt.close()

        for n_algo in range(len(algorithms)):
            mean_rewards[n_algo] = np.mean(mean_rewards[n_algo])
        return mean_rewards
        # /---- standard ----

    def sample_initial_state(self):
        init_observation = np.maximum(np.random.normal(loc=self.steady_observations, scale=self.initial_state_scale), 0, dtype=self.np_dtype)
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
