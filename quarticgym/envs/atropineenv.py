import numpy as np
from casadi import DM
import matplotlib.pyplot as plt
from gym import spaces, Env


from .helpers.helper_funcs import state_estimator
from .models.atropine_process import Plant
from .models.config import ND1, ND2, ND3, V1, V2, V3, V4, dt
from .helpers.constants import USS, INPUT_REFS, OUTPUT_REF, SIM_TIME
from .utils import *

class AtropineEnvGym(Env):
    def __init__(self, normalize=True, max_steps: int=60, x0_loc='quarticgym/datasets/atropineenv/x0.txt', z0_loc='quarticgym/datasets/atropineenv/z0.txt', model_loc='quarticgym/datasets/atropineenv/model.npy', uss_subtracted=False, reward_on_ess_subtracted=False, reward_on_steady=True, reward_on_absolute_efactor=False, reward_on_actions_penalty=0.0, reward_on_reject_actions=True, relaxed_max_min_actions=False, observation_include_t=True, observation_include_action=False, observation_include_uss=True, observation_include_ess=True, observation_include_e=True, observation_include_kf=True, observation_include_z=True, observation_include_x=False):
        self.normalize = normalize
        self.max_steps = max_steps # how many steps can this env run. if self.max_steps == -1 then run forever.
        self.action_dim = 4
        self.uss_subtracted = uss_subtracted # we assume that we can see the steady state output during steps. If true, we plus the actions with USS during steps.
        self.reward_on_ess_subtracted = reward_on_ess_subtracted
        self.reward_on_steady = reward_on_steady # whether reward base on Efactor (the small the better) or base on how close it is to the steady e-factor
        self.reward_on_absolute_efactor = reward_on_absolute_efactor # whether reward base on absolute Efactor. (is a valid input only if reward_on_steady is False)
        self.reward_on_actions_penalty = reward_on_actions_penalty
        self.reward_on_reject_actions = reward_on_reject_actions # when input actions are larger than max_actions, reject it and end the env immediately. 
        self.relaxed_max_min_actions = relaxed_max_min_actions # assume uss_subtracted = false.

        # now, select what to include during observations. by default we should have format like 
        # USS1, USS2, USS3, USS4, U1, U2, U3, U4, ESS, E, KF_X1, KF_X2, Z1, Z2, ..., Z30
        self.observation_include_t = observation_include_t # 1
        self.observation_include_action = observation_include_action # 4
        self.observation_include_uss = observation_include_uss # 4
        self.observation_include_ess = observation_include_ess # yss, 1
        self.observation_include_e = observation_include_e # y, efactor, 1
        self.observation_include_kf = observation_include_kf # after kalman filter, 2
        self.observation_include_z = observation_include_z # 30
        self.observation_include_x = observation_include_x # 1694

        if type(x0_loc) is str:
            self.x0 = np.loadtxt(x0_loc)  # initial states [0,50]
        elif type(x0_loc) is np.ndarray:
            self.x0 = x0_loc
        elif type(x0_loc) is list:
            self.x0 = np.array(x0_loc)
        else:
            raise Exception("x0_loc must be a string, list or a numpy array")
        if type(z0_loc) is str:
            self.z0 = np.loadtxt(z0_loc)  # initial states [0,50]
        elif type(z0_loc) is np.ndarray:
            self.z0 = z0_loc
        elif type(z0_loc) is list:
            self.z0 = np.array(z0_loc)
        else:
            raise Exception("z0_loc must be a string, list or a numpy array")
        self.model_preconfig = np.load(model_loc, allow_pickle=True)  # model

        # for a fixed batch.
        self.ur = INPUT_REFS  # reference inputs
        self.yr = OUTPUT_REF  # reference output
        self.num_sim = int(SIM_TIME / dt) # SIM_TIME/ 400 hours as fixed batch.
                
        self.observation_dim = 1 * self.observation_include_t + 4 * self.observation_include_action + 4 * self.observation_include_uss + 1 * self.observation_include_ess + \
            1 * self.observation_include_e + 2 * self.observation_include_kf + 30 * self.observation_include_z + 1694 * self.observation_include_x
        max_observations = []
        if self.observation_include_t:
            max_observations.append(np.ones(1, dtype=np.float32)*100.0) # by convention
        if self.observation_include_action:
            max_observations.append(np.ones(4, dtype=np.float32)*1.0) # by convention
        if self.observation_include_uss:
            max_observations.append(np.ones(4, dtype=np.float32)*0.5) # from dataset
        if self.observation_include_ess:
            max_observations.append(np.ones(1, dtype=np.float32)*15.0) # from dataset
        if self.observation_include_e:
            max_observations.append(np.ones(1, dtype=np.float32)*20.0) # from dataset
        if self.observation_include_kf:
            max_observations.append(np.ones(2, dtype=np.float32)*0.05) # from dataset
        if self.observation_include_z:
            max_observations.append(np.ones(30, dtype=np.float32)*0.5) # by convention
        if self.observation_include_x:
            max_observations.append(np.ones(1694, dtype=np.float32)*50.0) # by convention

        try:
            self.max_observations = np.concatenate(max_observations)
        except ValueError:
            raise Exception("observations must contain something! Need at least one array to concatenate")
        self.min_observations = np.zeros(self.observation_dim, dtype=np.float32)
        if not self.uss_subtracted:
            self.max_actions = np.array([0.408, 0.125, 0.392, 0.214], dtype=np.float32) # from dataset
            self.min_actions = np.array([0.4075, 0.105, 0.387, 0.208], dtype=np.float32) # from dataset
            if self.relaxed_max_min_actions:
                self.max_actions = np.array([0.5, 0.2, 0.5, 0.4], dtype=np.float32) # from dataset
                self.min_actions = np.array([0.3, 0.0, 0.2, 0.1], dtype=np.float32) # from dataset
        else:
            self.max_actions = np.array([1.92476206e-05, 1.22118426e-02, 1.82154982e-03, 3.59729230e-04], dtype=np.float32) # from dataset
            self.min_actions = np.array([-0.00015742, -0.00146234, -0.00021812, -0.00300454], dtype=np.float32) # from dataset
            if self.relaxed_max_min_actions:
                self.max_actions = np.array([2.0e-05, 1.3e-02, 2.0e-03, 4.0e-04], dtype=np.float32) # from dataset
                self.min_actions = np.array([-0.00016, -0.0015, -0.00022, -0.00301], dtype=np.float32) # from dataset
        if self.normalize:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(self.observation_dim,))
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        else:
            self.observation_space = spaces.Box(low=self.min_observations, high=self.max_observations, shape=(self.observation_dim,))
            self.action_space = spaces.Box(low=self.min_actions, high=self.max_actions, shape=(self.action_dim,))
        self.plant = Plant(ND1, ND2, ND3, V1, V2, V3, V4, dt)
        self.yss = self.plant.calculate_Efactor(DM(self.z0))  # steady state output, e-factor

    def reset(self):
        self.U = []  # inputs
        self.Y = []
        self.zk = DM(self.z0) # 30
        self.xk = self.plant.mix_and_get_initial_condition(self.x0, USS)[0] # 1694
        self.t = 0
        self.previous_efactor = self.yss # steady state output, e-factor, the small the better
        observations = []
        if self.observation_include_t:
            observations.append(np.array([self.t], dtype=np.float32))
        if self.observation_include_action:
            observations.append(np.zeros(4, dtype=np.float32))
        if self.observation_include_uss:
            observations.append(USS)
        if self.observation_include_ess:
            observations.append(np.array([self.yss], dtype=np.float32))
        if self.observation_include_e:
            observations.append(np.array([self.previous_efactor], dtype=np.float32))
        if self.observation_include_kf:
            self.xe = np.zeros(2, dtype=np.float32)
            observations.append(self.xe)
        if self.observation_include_z:
            observations.append(self.zk.full().flatten())
        if self.observation_include_x:
            observations.append(self.xk.full().flatten())
        try:
            observation = np.concatenate(observations)
        except ValueError:
            raise Exception("observations must contain something! Need at least one array to concatenate")
        if self.normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        
        return observation

    def _step(self, action):
        if self.max_steps == -1:
            done = False
        else:
            done = (self.t >= self.max_steps - 1)
        action = np.array(action, dtype=np.float32)
        if self.normalize:
            action, _, _ = denormalize_spaces(action, self.max_actions, self.min_actions)
            
        if self.reward_on_reject_actions: 
            if (action > self.max_actions).any() or (action < self.min_actions).any():
                reward = -100000.0
                done = True
                observation = np.zeros(self.observation_dim, dtype=np.float32)
                return observation, reward, done, {"efactor": 100000.0, "previous_efactor": self.previous_efactor, "reward_on_steady": reward, "reward_on_absolute_efactor": reward, "reward_on_efactor_diff": reward}

        if self.uss_subtracted:
            uk = [
                    action[0] + USS[0],
                    action[1] + USS[1],
                    action[2] + USS[2],
                    action[3] + USS[3]
                ]
        else:
            uk = action
        self.U.append(uk)
        _, xnext, znext = self.plant.simulate(self.xk, self.zk, uk)
        efactor = self.plant.calculate_Efactor(znext)
        self.Y.append(efactor)
        reward_on_steady = -abs(efactor - self.yss)
        reward_on_absolute_efactor = -abs(efactor)
        reward_on_efactor_diff = self.previous_efactor - efactor
        previous_efactor = self.previous_efactor
        if self.reward_on_ess_subtracted:
            reward = self.yss - efactor # efactor the smaller the better
        elif self.reward_on_steady:
            reward = reward_on_steady
        else:
            if self.reward_on_absolute_efactor:
                reward = reward_on_absolute_efactor
            else:
                reward = reward_on_efactor_diff
        reward += np.linalg.norm(action*self.reward_on_actions_penalty, ord=2)
        self.previous_efactor = efactor
        self.xk = xnext
        self.zk = znext
        observations = []
        if self.observation_include_t:
            observations.append(np.array([self.t], dtype=np.float32))
        if self.observation_include_action:
            observations.append(action)
        if self.observation_include_uss:
            observations.append(USS)
        if self.observation_include_ess:
            observations.append(np.array([self.yss], dtype=np.float32))
        if self.observation_include_e:
            observations.append(np.array([self.previous_efactor], dtype=np.float32)) #!!!!!!!also, shall I run a step here? how do we align?
        if self.observation_include_kf:
            xe = state_estimator(
                self.xe , uk, efactor - self.yss, #self.Xhat[k] is previous step xe
                self.model_preconfig[0], self.model_preconfig[1],
                self.model_preconfig[2], self.model_preconfig[4]
            )
            observations.append(xe)
        if self.observation_include_z:
            observations.append(self.zk.full().flatten())
        if self.observation_include_x:
            observations.append(self.xk.full().flatten())
        try:
            observation = np.concatenate(observations)
        except ValueError:
            raise Exception("observations must contain something! Need at least one array to concatenate")
        if self.normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        self.t += 1
        return observation, reward, done, {"efactor": efactor, "previous_efactor": previous_efactor, "reward_on_steady": reward_on_steady, "reward_on_absolute_efactor": reward_on_absolute_efactor, "reward_on_efactor_diff": reward_on_efactor_diff}
        # state, reward, done, info in gym env term 

    def step(self, action):
        try:
            return self._step(action)
        except Exception:
            reward = -100000.0
            done = True
            observation = np.zeros(self.observation_dim, dtype=np.float32)
            return observation, reward, done, {"efactor": 100000.0, "previous_efactor": self.previous_efactor, "reward_on_steady": reward, "reward_on_absolute_efactor": reward, "reward_on_efactor_diff": reward}

    def plot(self, show=False, efactor_fig_name=None, input_fig_name=None):
        target_efactor = [self.yss + self.yr] * self.num_sim
        target_inputs = [USS + self.ur] * self.num_sim
        U = np.array(self.U) * 1000  # scale the solution to micro Litres
        target_inputs = np.array(target_inputs) * 1000  # scale the solution to micro Litres
        local_t = [k * dt for k in range(self.num_sim)]
        # plots
        plt.close("all")
        plt.figure(0)
        plt.plot(local_t, self.Y, label='Real Output')
        plt.plot(local_t, target_efactor, linestyle="--", label='Steady State Output')
        plt.xlabel('Time [min]')
        plt.ylabel('E-Factor [A.U.]')
        plt.legend()
        plt.grid()
        if efactor_fig_name is not None:
            plt.savefig(efactor_fig_name)
        plt.tight_layout()

        # create figure (fig), and array of axes (ax)
        fig, axs = plt.subplots(nrows=2, ncols=2)
        axs[0, 0].step(local_t, U[:, 0], where='post', label='Real Input')
        axs[0, 0].step(local_t, target_inputs[:, 0], where='post', linestyle="--", label='Steady State Input')
        axs[0, 0].set_ylabel(u'U1 [\u03bcL/min]')
        axs[0, 0].set_xlabel('time [min]')
        axs[0, 0].grid()

        axs[0, 1].step(local_t, U[:, 1], where='post', label='Real Input')
        axs[0, 1].step(local_t, target_inputs[:, 1], where='post', linestyle="--", label='Steady State Input')
        axs[0, 1].set_ylabel(u'U2 [\u03bcL/min]')
        axs[0, 1].set_xlabel('time [min]')
        axs[0, 1].grid()

        axs[1, 0].step(local_t, U[:, 2], where='post', label='Real Input')
        axs[1, 0].step(local_t, target_inputs[:, 2], where='post', linestyle="--", label='Steady State Input')
        axs[1, 0].set_ylabel(u'U3 [\u03bcL/min]')
        axs[1, 0].set_xlabel('time [min]')
        axs[1, 0].grid()

        axs[1, 1].step(local_t, U[:, 3], where='post', label='Real Input')
        axs[1, 1].step(local_t, target_inputs[:, 3], where='post', linestyle="--", label='Steady State Input')
        axs[1, 1].set_ylabel(u'U4 [\u03bcL/min]')
        axs[1, 1].set_xlabel('time [min]')
        axs[1, 1].legend()
        plt.tight_layout()
        plt.grid()
        if input_fig_name is not None:
            plt.savefig(input_fig_name)
        if show:
            plt.show()
        else:
            plt.close()
