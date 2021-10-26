import mpctools as mpc      # import mpctools: https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/
import numpy as np
from gym import spaces, Env
from .utils import *


MAX_OBSERVATIONS = [1.0, 100.0, 1.0] # cA, T, h
MIN_OBSERVATIONS = [0.0, 0.0, 0.0]
MAX_ACTIONS = [100.0, 0.2] #Tc, qout
MIN_ACTIONS = [0.0, 0.0]
ERROR_REWARD = -100.0


class ReactorModel:
    
    def __init__(self, sampling_time):
        
        # Define model parameters
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
            self.q_in*(self.cAf - c)/(np.pi*self.r**2*h) - rate, # kmol/m^3/min
            self.q_in*(self.Tf - T)/(np.pi*self.r**2*h) 
                        - self.dH/(self.rho*self.Cp)*rate
                        + 2*self.U/(self.r*self.rho*self.Cp)*(Tc - T), # degree C/min
            (self.q_in - q)/(np.pi*self.r**2)     # m/min
                ]
        return dxdt
    
    # builds a reactor using mpctools and casadi
    def build_reactor_simulator(self):
        self.simulator = mpc.DiscreteSimulator(self.ode, self.sampling_time, [self.Nx, self.Nu], ["x", "u"])
    
    # integrates one sampling time or time step and returns the next state
    def step(self, x, u):
        return self.simulator.sim(x,u)


class ReactorEnv(Env):

    def __init__(self, dense_reward=True, normalize=True, action_dim=2, observation_dim=3, reward_function=None, done_calculator=None, # general env inputs
    sampling_time=0.1, max_steps=100):
        # ---- standard ----
        self.step_count = 0
        self.total_reward = 0
        self.done = False
        self.dense_reward = dense_reward
        self.normalize = normalize
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.reward_function = reward_function
        self.done_calculator = done_calculator
        if self.reward_function is None:
            self.reward_function = self.reward_function_standard
        if self.done_calculator is None:
            self.done_calculator = self.done_calculator_standard
        # ---- standard ----

        self.sampling_time = sampling_time
        self.max_steps = max_steps

        # ---- standard ----
        self.max_observations = np.array(MAX_OBSERVATIONS, dtype=np.float32)
        self.min_observations = np.array(MIN_OBSERVATIONS, dtype=np.float32)
        self.max_actions = np.array(MAX_ACTIONS, dtype=np.float32)
        self.min_actions = np.array(MIN_ACTIONS, dtype=np.float32)
        if self.normalize:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(self.observation_dim,))
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        else:
            self.observation_space = spaces.Box(low=self.min_observations, high=self.max_observations, shape=(self.observation_dim,))
            self.action_space = spaces.Box(low=self.min_actions, high=self.max_actions, shape=(self.action_dim,))
        # ---- standard ----
        
        self.steady_observations = np.array([0.8778252, 51.34660837, 0.659], dtype=np.float32)
        self.steady_actions = np.array([26.85, 0.1], dtype=np.float32) 
        
    def sample_initial_state(self):
        init_observation = np.maximum(np.random.normal(loc=[0.8778252, 51.34660837, 0.659], scale=[0.25, 25, 0.2]), 0, dtype=np.float32)
        init_observation = init_observation.clip(self.min_observations, self.max_observations)
        return init_observation

    def observation_beyond_box(self, observation):
        return np.any(observation > self.max_observations) or np.any(observation < self.min_observations)

    def reward_function_standard(self, previous_observation, action, current_observation, reward=None):
        # ---- standard ----
        # s, a, r, s, a
        if reward is not None:
            return reward
        elif self.observation_beyond_box(current_observation):
            return ERROR_REWARD
        # ---- standard ----
        reward = - ( np.mean((current_observation - self.steady_observations) ** 2 / np.maximum((self.init_observation - self.steady_observations) ** 2, 1e-8)) )
        return reward
    
    def done_calculator_standard(self, current_observation, step_count, done=None):
        # ---- standard ----
        if done is not None:
            return done
        elif self.observation_beyond_box(current_observation):
            return True
        # ---- standard ----
        if step_count >= self.max_steps: # same as range(0, max_steps)
            return True
        else:
            return False

    def reset(self, initial_state=None):
        # ---- standard ----
        self.step_count = 0
        self.total_reward = 0
        self.done = False
        
        if initial_state is not None:
            initial_state = np.array(initial_state, dtype=np.float32)
            observation = initial_state
            self.init_observation = initial_state
        else:
            observation = self.sample_initial_state()
            self.init_observation = observation
        # ---- standard ----
        self.previous_observation = observation
        self.reactor = ReactorModel(self.sampling_time)
        self.reactor.build_reactor_simulator()
        # self.x = np.zeros((self.max_steps+1, self.observation_dim))
        # self.x[0, :] = observation
        # self.u = np.zeros((self.max_steps, self.action_dim))

        # ---- standard ----
        if self.normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        return observation
        # ---- standard ----

    def step(self, action):
        # ---- standard ----
        action = np.array(action, dtype=np.float32)
        if self.normalize:
            action, _, _ = denormalize_spaces(action, self.max_actions, self.min_actions)
        # ---- standard ----
        try:
            observation = self.reactor.step(self.previous_observation, action)
        except:
            # may encounter casadi error here.
            observation = self.previous_observation
            reward = ERROR_REWARD
            done = True

        # ---- standard ----
        # compute reward
        reward = self.reward_function(self.previous_observation, action, observation)
        # compute done
        done = self.done_calculator(observation, self.step_count)
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
        if self.normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        self.step_count += 1
        return observation, reward, done, {}
        # ---- standard ----
