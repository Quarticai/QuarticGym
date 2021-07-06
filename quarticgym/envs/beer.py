import numpy as np
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# T range 9-16
# biomass -> 0 or dont move means episode end, reward every step -1
# for i in range 1000?

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


# Case1: profile with changed temp profile
res_industrial = []
profile_industrial = [11 + 1 / 8 * i for i in range(25)] \
                          + [14] * 95 \
                          + [14 + 2 / 25 * i for i in range(25)] \
                          + [16] * 25 + [16 - 8 / 15 * i for i in range(30)]

for i in range(200):
    if i == 0:
        X_A, X_L, X_D, S, EtOH, DY, EA = 0, 2, 2, 130, 0, 0, 0
    t = np.arange(0 + i, 1 + i, 0.01)
    sol = odeint(beer_ode, (X_A, X_L, X_D, S, EtOH, DY, EA), t, args=([130, profile_industrial[i] + 273.15],))
    X_A, X_L, X_D, S, EtOH, DY, EA = sol[-1, :] # observation/state
    # profile_industrial[i] + 273.15 means tempreture, T
    # X_A+X_L+X_D < 0.5 means end
    # X_A+X_L+X_D -> 0 fast, S needs to go to zero, EtOH > 50, the more the better, reward 1:1:1
    res_industrial.append(sol[-1, :])
res_industrial = np.array(res_industrial)

# Case2: profile with constant temp profile
res_cons = []
profile_cons = [13] * 200

for i in range(200):
    if i == 0:
        X_A, X_L, X_D, S, EtOH, DY, EA = 0, 2, 2, 130, 0, 0, 0
    t = np.arange(0 + i, 1 + i, 0.01)
    sol = odeint(beer_ode, (X_A, X_L, X_D, S, EtOH, DY, EA), t, args=([130, profile_cons[i] + 273.15],))
    X_A, X_L, X_D, S, EtOH, DY, EA = sol[-1, :]
    res_cons.append(sol[-1, :])
res_cons = np.array(res_cons)


# plots
plt.subplot(2, 2, 1)
plt.plot(profile_industrial, label='Industrial', color='blue')
plt.plot(profile_cons, label='Isothermal', color='blue', linestyle='dashed')
plt.autoscale(enable=True, axis='both', tight=True)
plt.ylim((0, 18))
plt.xlabel('Time [h]')
plt.ylabel('Temperature [\u00B0C]')
plt.title("Fermentation Profile")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(res_industrial[:, 3], label='Sugar', color='m')
plt.plot(res_cons[:, 3], color='m', linestyle='dashed')
plt.plot(res_industrial[:, 4], label='Ethanol', color='orange')
plt.plot(res_cons[:, 4], color='orange', linestyle='dashed')
plt.autoscale(enable=True, axis='both', tight=True)
plt.ylim((0, 140))
plt.xlabel('Time [h]')
plt.ylabel('Concentration [g/L]')
plt.title("Sugar and Ethanol Production")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(res_industrial[:, 0], label='Active', color='green')
plt.plot(res_cons[:, 0], color='green', linestyle='dashed')
plt.plot(res_industrial[:, 1], label='Lag', color='c')
plt.plot(res_cons[:, 1], color='c', linestyle='dashed')
plt.plot(res_industrial[:, 2], label='Dead', color='red')
plt.plot(res_cons[:, 2], color='red', linestyle='dashed')
plt.plot(res_industrial[:, 0] + res_industrial[:, 1] + res_industrial[:, 2], label='Total', color='black')
plt.plot(res_cons[:, 0] + res_cons[:, 1] + res_cons[:, 2], color='black', linestyle='dashed')
plt.autoscale(enable=True, axis='both', tight=True)
plt.ylim((0, 9))
plt.xlabel('Time [h]')
plt.ylabel('Concentration [g/L]')
plt.title("Biomass Evolution")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(res_industrial[:, 5], label='Diacetyl', color='darkgoldenrod')
plt.plot(res_cons[:, 5], color='darkgoldenrod', linestyle='dashed')
plt.plot(res_industrial[:, 6], label='Ethyl Acelate', color='grey')
plt.plot(res_cons[:, 6], color='grey', linestyle='dashed')
plt.autoscale(enable=True, axis='both', tight=True)
plt.ylim((0, 1.6))
plt.xlabel('Time [h]')
plt.ylabel('Concentration [ppm]')
plt.title("By-Product Production")
plt.legend()
plt.tight_layout()
plt.show()