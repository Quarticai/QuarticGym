import numpy as np
import math
from scipy.integrate import odeint
import random
from .utils import *
from gym import spaces, Env


random.seed(0)
MAX_LENGTH = 200
INIT_SUGAR = 130
BEER_init = [0, 2, 2, INIT_SUGAR, 0, 0, 0, 0] # X_A, X_L, X_D, S, EtOH, DY, EA = 0, 2, 2, 130, 0, 0, 0
BEER_min = [0, 0, 0, 0, 0, 0, 0, 0]
BEER_max = [15, 15, 15, 150, 150, 10, 10, MAX_LENGTH]
TEMPERATURE_min = [9.0]
TEMPERATURE_max = [16.0]
BIOMASS_end_threshold = 0.5
BIOMASS_end_change_threshold = 0.01
SUGAR_end_threshold = 0.5


def beer_ode(points, t, sets):
    """
    Beer fermentation process
    """
    X_A, X_L, X_D, S, EtOH, DY, EA = points
    S0, T = sets
    k_x = 0.5 * S0

    u_x0 = math.exp(108.31 - 31934.09 / T)
    Y_EA = math.exp(89.92 - 26589 / T)
    u_s0 = math.exp(-41.92 + 11654.64 / T)
    u_L = math.exp(30.72 - 9501.54 / T)

    u_DY = 0.000127672
    u_AB = 0.00113864

    u_DT = math.exp(130.16 - 38313 / T)
    u_SD0 = math.exp(33.82 - 10033.28 / T)
    u_e0 = math.exp(3.27 - 1267.24 / T)
    k_s = math.exp(-119.63 + 34203.95 / T)

    u_x = u_x0 * S / (k_x + EtOH)
    u_SD = u_SD0 * 0.5 * S0 / (0.5 * S0 + EtOH)
    u_s = u_s0 * S / (k_s + S)
    u_e = u_e0 * S / (k_s + S)
    f = 1 - EtOH / (0.5 * S0)

    dXAt = u_x * X_A - u_DT * X_A + u_L * X_L
    dXLt = -u_L * X_L
    dXDt = -u_SD * X_D + u_DT * X_A
    dSt = -u_s * X_A
    dEtOHt = f * u_e * X_A
    dDYt = u_DY * S * X_A - u_AB * DY * EtOH
    # todo
    dEAt = Y_EA * u_x * X_A
    # dEAt = -Y_EA * u_s * X_A

    return np.array([dXAt, dXLt, dXDt, dSt, dEtOHt, dDYt, dEAt])


class BeerFMTEnvGym(Env):

    def __init__(self, dense_reward=True, normalize=True, observation_relaxation=1.0, action_dim=1, observation_dim=8):
        """
        Time is in our observation_space. We make the env time aware.
        The only action/input is temperature.
        The observations are X_A, X_L, X_D, S, EtOH, DY, EA, time
        """
        self.dense_reward = dense_reward
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.observation_space = spaces.Box(low=-1*observation_relaxation, high=1*observation_relaxation, shape=(self.observation_dim,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        # ---- set by dataset or use predefined as you wish if applicable ----
        self.normalize = normalize
        self.max_observations = np.array(BEER_max, dtype=np.float32)
        self.min_observations = np.array(BEER_min, dtype=np.float32)
        self.max_actions = np.array(TEMPERATURE_max, dtype=np.float32)
        self.min_actions = np.array(TEMPERATURE_min, dtype=np.float32)
        # ---- set by dataset or use predefined as you wish if applicable ----
        self.res_forplot = [] # for plotting purposes

    def reaction_finish_calculator(self, X_A, X_L, X_D, S, EtOH, DY, EA):
        # X_A+X_L+X_D < 0.5 means end
        # X_A+X_L+X_D -> 0 fast, S needs to go to zero, EtOH > 50, the more the better, reward 1:1:1
        # T range 9-16
        # biomass -> 0 or dont move means episode end, reward every step -1
        finished = False
        current_biomass = X_A+X_L+X_D
        if current_biomass < BIOMASS_end_threshold or abs(current_biomass - self.prev_biomass) < BIOMASS_end_change_threshold:
            if S < SUGAR_end_threshold:
                if EtOH > 50.0:
                    finished = True

        self.prev_biomass = current_biomass
        return finished

    def reset(self):
        self.time = 0
        self.total_reward = 0
        self.done = False
        observation = BEER_init
        self.prev_biomass = observation[0]+observation[1]+observation[2]
        observation = np.array(observation, dtype=np.float32)
        self.prev_denormalized_observation = observation
        if self.normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        
        return observation
    
    def step(self, action):
        action = np.array(action, dtype=np.float32)
        if self.normalize:
            action, _, _ = denormalize_spaces(action, self.max_actions, self.min_actions)
        t = np.arange(0 + self.time, 1 + self.time, 0.01)
        X_A, X_L, X_D, S, EtOH, DY, EA, _ = self.prev_denormalized_observation
        sol = odeint(beer_ode, (X_A, X_L, X_D, S, EtOH, DY, EA), t, args=([INIT_SUGAR, action[0] + 273.15],))
        self.res_forplot.append(sol[-1, :])
        self.time += 1
        X_A, X_L, X_D, S, EtOH, DY, EA = sol[-1, :]
        observation = [X_A, X_L, X_D, S, EtOH, DY, EA, self.time]
        observation = np.array(observation, dtype=np.float32)
        self.prev_denormalized_observation = observation
        finished = self.reaction_finish_calculator(X_A, X_L, X_D, S, EtOH, DY, EA)
        done = (finished or self.time == MAX_LENGTH)
        if finished:
            reward = 200
        elif done:        # reaches time limit but reaction has not finished
            reward = -200
        else:
            reward = -1   # we want the simulation to end fast
        self.total_reward += reward

        if self.dense_reward:
            reward = reward # conventional
        elif not done:
            reward = 0.0
        else:
            reward = self.total_reward
        if self.normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        return observation, reward, done, {"res_forplot": np.array(self.res_forplot)}
