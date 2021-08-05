import numpy as np
import scipy.io as sio
from casadi import DM
from scipy import signal
from typing import List
from tqdm.auto import tqdm
import control as ctrl
import matplotlib.pyplot as plt
from gym import spaces, Env


from .helpers.helper_funcs import fitting_score, state_estimator
from .helpers.mpc_controller import mpc_controller
from .models.atropine_process import Plant
from .models.config import ND1, ND2, ND3, V1, V2, V3, V4, dt
from .helpers.constants import USS, INPUT_REFS, OUTPUT_REF, TRAIN_SIZE, SIM_TIME
from .SIPPY.sippy.OLSims_methods import SS_model
from .SIPPY.sippy import functionsetSIM as fsetSIM, system_identification
from .utils import *


class AtropineEnvGym(Env):
    def __init__(self, normalize=True, data_len: int=200, x0_loc='quarticgym/datasets/atropineenv/x0.txt', z0_loc='quarticgym/datasets/atropineenv/z0.txt', u_loc ='quarticgym/datasets/atropineenv/U.mat', uss_observable=False, reward_on_steady=True):
        self.normalize = normalize
        self.data_len = data_len
        self.action_dim = 4
        self.observation_dim = 1724
        self.uss_observable = uss_observable # we assume that we can see the steady state output during steps.
        self.reward_on_steady = reward_on_steady # whether reward base on Efactor (the small the better) or base on how close it is to the steady e-factor
        self.x0 = np.loadtxt(x0_loc)  # initial states [0,50]
        self.z0 = np.loadtxt(z0_loc)  # initial algebraic states [0, 0.5]
        self.max_observations = np.concatenate([np.ones(1694, dtype=np.float32)*50, np.ones(30, dtype=np.float32)*0.5])
        self.min_observations = np.zeros(self.observation_dim, dtype=np.float32)
        self.max_actions = np.ones(4, dtype=np.float32) * 5
        self.min_actions = np.zeros(4, dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.observation_dim,))
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_dim,))
        self.plant = Plant(ND1, ND2, ND3, V1, V2, V3, V4, dt)
        self.yss = self.plant.calculate_Efactor(DM(self.z0))  # steady state output, e-factor
        self.u_loc = u_loc

    def reset(self):
        self.U = sio.loadmat(self.u_loc)['U'][:, :self.data_len]  # inputs
        self.Y = []
        self.zk = DM(self.z0) # 30
        self.xk = self.plant.mix_and_get_initial_condition(self.x0, USS)[0] # 1694
        self.t = 0
        self.previous_efactor = self.yss # steady state output, e-factor, the small the better
        observation = np.concatenate((self.xk.full(), self.zk.full()), axis=0).flatten()
        if self.normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)

        return observation

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        if self.normalize:
            action, _, _ = denormalize_spaces(action, self.max_actions, self.min_actions)
        if self.uss_observable:
            uk = [
                    action[0] + USS[0],
                    action[1] + USS[1],
                    action[2] + USS[2],
                    action[3] + USS[3]
                ]
        else:
            uk = action
        _, xnext, znext = self.plant.simulate(self.xk, self.zk, uk)
        efactor = self.plant.calculate_Efactor(znext)
        if self.reward_on_steady:
            reward = -abs(efactor - self.yss)
        else:
            reward = self.previous_efactor - efactor
            self.previous_efactor = efactor
        self.xk = xnext
        self.zk = znext
        observation = np.concatenate((self.xk.full(), self.zk.full()), axis=0).flatten()
        if self.normalize:
            observation, _, _ = normalize_spaces(observation, self.max_observations, self.min_observations)
        self.t += 1
        return observation, reward, False, {}
        # state, reward, done, info in gym env term 


class OpenLoop:
    def __init__(
        self,
        data_len: int
    ):
        action_dim = 4


        self.x0 = np.loadtxt('quarticgym/datasets/atropineenv/x0.txt')  # initial states
        self.z0 = np.loadtxt('quarticgym/datasets/atropineenv/z0.txt')  # initial algebraic states
        self.U = sio.loadmat('quarticgym/datasets/atropineenv/U.mat')['U'][:, :data_len]  # inputs
        self.plant = Plant(ND1, ND2, ND3, V1, V2, V3, V4, dt)  # plant model
        self.yss = self.plant.calculate_Efactor(DM(self.z0))  # steady state output, e-factor
        self.Y = []

    def run(self):
        zk = DM(self.z0) # 30
        xk = self.plant.mix_and_get_initial_condition(self.x0, USS)[0] # 1694
        for k in tqdm(range(len(self.U[0]))):
            uk = [
                self.U[0][k] + USS[0],
                self.U[1][k] + USS[1],
                self.U[2][k] + USS[2],
                self.U[3][k] + USS[3]
            ]
            _, xnext, znext = self.plant.simulate(xk, zk, uk)
            efactor = self.plant.calculate_Efactor(znext) # float
            self.Y.append(efactor)
            xk, zk = xnext, znext


class SystemIdentifier:
    def __init__(
        self,
        yss: float,
        U: np.ndarray,
        Y: List
    ):
        self.yss = yss  # steady state output, e-factor
        self.U = U  # inputs
        self.Y = Y  # outputs

        self.U_train = None
        self.Y_train = None
        self.U_test = None
        self.Y_test = None

        self.model = None

    def _split(self):
        len_train = round(TRAIN_SIZE * len(self.U[0]))
        self.Y = signal.detrend(self.Y)
        self.U_train = self.U[:, :len_train]
        self.Y_train = self.Y[:len_train]
        self.U_test = self.U[:, len_train:]
        self.Y_test = self.Y[len_train:]

    def run(self):
        self._split()
        self.model = system_identification(
            self.Y_train,
            self.U_train,
            'N4SID',
            SS_fixed_order=2
        )
        xid, yid = fsetSIM.SS_lsim_process_form(
            self.model.A,
            self.model.B,
            self.model.C,
            self.model.D,
            self.U_train
        )
        score = fitting_score(self.Y_train, yid)
        print(f"=== The training score is {score}")

        # starting x0 from the training data
        x0 = xid[:, -1].reshape((1, len(xid[:, -1])))
        xid, yid = fsetSIM.SS_lsim_process_form(
            self.model.A,
            self.model.B,
            self.model.C,
            self.model.D,
            self.U_test,
            x0
        )
        score = fitting_score(self.Y_test, yid)
        print(f"=== The testing score is {score}")


class ControlSystem:
    def __init__(
        self,
        model: SS_model,
        yss: float
    ):
        self.x0 = np.loadtxt('quarticgym/datasets/atropineenv/x0u.txt')  # initial states
        self.z0 = np.loadtxt('quarticgym/datasets/atropineenv/z0u.txt')  # initial algebraic states
        self.ur = INPUT_REFS  # reference inputs
        self.yr = OUTPUT_REF  # reference output
        self.yss = yss  # steady state output
        self.model = model
        self._check(
            A=self.model.A,
            C=self.model.C
        )
        self.plant = Plant(ND1, ND2, ND3, V1, V2, V3, V4, dt)

        self.N = 30  # prediction and control horizon
        self.Nx = len(model.A)  # dimension of state
        self.Nu = 4  # dimension of input

        self.X = [self.plant.mix_and_get_initial_condition(self.x0, USS)[0]]
        self.Z = [DM(self.z0)]
        self.Xhat = [np.ones(self.Nx) * 0.001]

        self.Y = []
        self.U = []
        self.num_sim = None
        self.t = []

    def _check(
        self,
        A: np.ndarray,
        C: np.ndarray
    ):
        # check observability to ensure state that the states can be estimated.
        # do not continue if it fails the test. model need to be re-identified.
        O = ctrl.obsv(A, C)
        assert (len(A) == np.linalg.matrix_rank(O))

    def run(self):
        self.num_sim = int(SIM_TIME / dt)
        for k in tqdm(range(self.num_sim)):
            self.t.append(k * dt)
            uk = mpc_controller(
                self.Xhat[k],
                self.N, self.Nx, self.Nu,
                USS, self.ur, self.yr,
                self.model.A, self.model.B,
                self.model.C, self.model.D) / 1000  # unscale

            # true plant
            # inputs from controller are in deviation form. add steady state value to get actual input
            x0, xk, zk = self.plant.simulate(self.X[k], self.Z[k], uk + USS)
            self.X[k][:] = x0[:]
            efactor = self.plant.calculate_Efactor(zk)

            # state estimator
            xe = state_estimator(
                self.Xhat[k], uk, efactor - self.yss,
                self.model.A, self.model.B,
                self.model.C, self.model.K
            )

            # record
            self.Xhat.append(xe)
            self.X.append(xk)
            self.Z.append(zk)

            self.U.append(uk + USS)
            self.Y.append(efactor)

        self._plot()

    def _plot(self):
        target_efactor = [self.yss + self.yr] * self.num_sim
        target_inputs = [USS + self.ur] * self.num_sim
        U = np.array(self.U) * 1000  # scale the solution to micro Litres
        target_inputs = np.array(target_inputs) * 1000  # scale the solution to micro Litres

        # plots
        plt.close("all")
        plt.figure(0)
        plt.plot(self.t, self.Y, label='Real Output')
        plt.plot(self.t, target_efactor, linestyle="--", label='Steady State Output')
        plt.xlabel('Time [min]')
        plt.ylabel('E-Factor [A.U.]')
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # create figure (fig), and array of axes (ax)
        fig, axs = plt.subplots(nrows=2, ncols=2)
        axs[0, 0].step(self.t, U[:, 0], where='post', label='Real Input')
        axs[0, 0].step(self.t, target_inputs[:, 0], where='post', linestyle="--", label='Steady State Input')
        axs[0, 0].set_ylabel(u'U1 [\u03bcL/min]')
        axs[0, 0].set_xlabel('time [min]')
        axs[0, 0].grid()

        axs[0, 1].step(self.t, U[:, 1], where='post', label='Real Input')
        axs[0, 1].step(self.t, target_inputs[:, 1], where='post', linestyle="--", label='Steady State Input')
        axs[0, 1].set_ylabel(u'U2 [\u03bcL/min]')
        axs[0, 1].set_xlabel('time [min]')
        axs[0, 1].grid()

        axs[1, 0].step(self.t, U[:, 2], where='post', label='Real Input')
        axs[1, 0].step(self.t, target_inputs[:, 2], where='post', linestyle="--", label='Steady State Input')
        axs[1, 0].set_ylabel(u'U3 [\u03bcL/min]')
        axs[1, 0].set_xlabel('time [min]')
        axs[1, 0].grid()

        axs[1, 1].step(self.t, U[:, 3], where='post', label='Real Input')
        axs[1, 1].step(self.t, target_inputs[:, 3], where='post', linestyle="--", label='Steady State Input')
        axs[1, 1].set_ylabel(u'U4 [\u03bcL/min]')
        axs[1, 1].set_xlabel('time [min]')
        axs[1, 1].legend()
        plt.tight_layout()
        plt.grid()
        plt.show()


# # Init an openloop simulation to get data
# openloop = OpenLoop(data_len=200)
# openloop.run()

# # Train a linear system
# system_identifier = SystemIdentifier(
#     yss=openloop.yss,
#     U=openloop.U,
#     Y=openloop.Y
# )
# system_identifier.run()

# # Apply the MPC control
# control_system = ControlSystem(
#     model=system_identifier.model,
#     yss=openloop.yss
# )
# control_system.run()
