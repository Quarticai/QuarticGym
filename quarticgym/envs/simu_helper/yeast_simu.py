import math
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


def sf_bioc_ode(points, t, sets):
    # Constants
    # Specific Ionic constants(l / g_ion)
    HNa = -0.550
    HCa = -0.303
    HMg = -0.314
    HH = -0.774
    HCl = 0.844
    HCO3 = 0.485
    HHO = 0.941

    # Molecular masses(g / mol)
    MNaCl = 58.5
    MCaCO3 = 90
    MMgCl2 = 95
    MNa = 23
    MCa = 40
    MMg = 24
    MCl = 35.5
    MCO3 = 60

    # Kinetic constants
    # miu_X = 0.556 # [1 / h]
    miu_P = 1.790  # [1 / h]
    Ks = 1.030  # [g / l]
    Ks1 = 1.680  # [g / l]
    Kp = 0.139  # [g / l]
    Kp1 = 0.070  # [g / l]
    Rsx = 0.607
    Rsp = 0.435
    YO2 = 0.970  # [mg / mg]
    KO2 = 8.86  # [mg / l]
    miu_O2 = 0.5  # [1 / h]
    A1 = 9.5e8
    A2 = 2.55e33
    Ea1 = 55000  # J / mol
    Ea2 = 220000  # J / mol
    R = 8.31  # J / (mol.K)

    # thermodynamic constants
    Kla0 = 38  # [1 / h]
    KT = 100 * 3600  # [J / hm2K]
    Vm = 50  # [l]
    AT = 1  # [m2]
    ro = 1080  # [g / l]
    ccal = 4.18  # [J / gK]
    roag = 1000  # [g / l]
    ccalag = 4.18  # [J / gK]
    deltaH = 518  # [kJ / mol O2 consumat]

    # Initial data
    mNaCl = 500  # [g]
    mCaCO3 = 100  # [g]
    mMgCl2 = 100  # [g]
    pH = 6
    Tiag = 15  # [°C]

    # if flag == 1:
    V = points[0]
    cX = points[1]
    cP = points[2]
    cS = points[3]
    cO2 = points[4]
    T = points[5]
    Tag = points[6]

    Fi = sets[0]  # l / h
    Fe = sets[1]  # l / h
    T_in = sets[2]  # K
    cS_in = sets[3]  # g / l
    Fag = sets[4]  # l / h

    c0st = 14.16 - 0.3943 * T + 0.007714 * T ** 2 - 0.0000646 * T ** 3  # [mg / l]
    cNa = mNaCl / MNaCl * MNa / V
    cCa = mCaCO3 / MCaCO3 * MCa / V
    cMg = mMgCl2 / MMgCl2 * MMg / V
    cCl = (mNaCl / MNaCl + 2 * mMgCl2 / MMgCl2) * MCl / V
    cCO3 = mCaCO3 / MCaCO3 * MCO3 / V
    cH = 10 ** (-pH)
    cOH = 10 ** (-(14 - pH))
    INa = 0.5 * cNa * ((+1) ** 2)
    ICa = 0.5 * cCa * ((+2) ** 2)
    IMg = 0.5 * cMg * ((+2) ** 2)
    ICl = 0.5 * cCl * ((-1) ** 2)
    ICO3 = 0.5 * cCO3 * ((-2) ** 2)
    IH = 0.5 * cH * ((+1) ** 2)
    IOH = 0.5 * cOH * ((-1) ** 2)
    sumaHiIi = HNa * INa + HCa * ICa + HMg * IMg + HCl * ICl + HCO3 * ICO3 + HH * IH + HHO * IOH
    cst = c0st * 10 ** (-sumaHiIi)
    alfa = 0.8
    Kla = Kla0 * (1.024 ** (T - 20))
    rO2 = miu_O2 * cO2 * cX / YO2 / (KO2 + cO2) * 1000  # mg / lh
    miu_X = A1 * math.exp(-Ea1 / R / (T + 273.15)) - A2 * math.exp(-Ea2 / R / (T + 273.15))

    dV = Fi - Fe
    dcX = miu_X * cX * cS / (Ks + cS) * math.exp(-Kp * cP) - (Fe / V) * cX  # g / (l.h)
    dcP = miu_P * cX * cS / (Ks1 + cS) * math.exp(-Kp1 * cP) - (Fe / V) * cP  # g / (l.h)
    dcS = - miu_X * cX * cS / (Ks + cS) * math.exp(-Kp * cP) / Rsx - miu_P * cX * cS / (Ks1 + cS) * math.exp(
        -Kp1 * cP) / Rsp + (Fi / V) * cS_in - (Fe / V) * cS  # g/(l.h)

    dcO2 = Kla * (cst - cO2) - rO2 - (Fe / V) * cO2  # mg / (l.h)
    dT = (1 / 32 * V * rO2 * deltaH - KT * AT * (T - Tag) + Fi * ro * ccal * (T_in + 273.15) - Fe * ro * ccal * (
        T + 273.15)) / (ro * ccal * V)  # J/h
    dTag = (Fag * ccalag * roag * (Tiag - Tag) + KT * AT * (T - Tag)) / (Vm * roag * ccalag)  # J/h

    return np.array([dV, dcX, dcP, dcS, dcO2, dT, dTag])


# Main simu loop
def get_yeast_batches(profile):
    batches = []
    for i in range(300):
        if i == 0:
            volume, yeast, ethanol, glucose, oxygen, T, Tag = [1000, 0.90467678228155, 12.51524128083789,
                                                               29.73892382828279, 3.10695341758232,
                                                               29.57321214183856, 27.05393890970931]
        t = np.arange(0 + i, 1 + i, 0.1)
        sol = odeint(sf_bioc_ode, (volume, yeast, ethanol, glucose, oxygen, T, Tag),
                     t,
                     args=([51, 51, 25, 60, profile[i]],))
        volume, yeast, ethanol, glucose, oxygen, T, Tag = sol[-1, :]
        batches.append(sol[-1, :])
    return np.array(batches)


# Pre-defined prifile
cooling_water_profile = [18] * 50 + np.linspace(18, 9, num=100).tolist() + [9] * 150
res_industrial = get_yeast_batches(cooling_water_profile)

print(res_industrial.shape)

plt.figure()
plt.plot(cooling_water_profile, color='red')
plt.autoscale(enable=True, axis='both', tight=True)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.xlabel('Time [h]')
plt.ylabel('cooling water flow rate [l/h]')
plt.title("Cooling water profile [l/h]")

plt.figure()
plt.subplot(3, 2, 1)
plt.plot(res_industrial[:, 0], color='blue')
plt.autoscale(enable=True, axis='both', tight=True)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.xlabel('Time [h]')
plt.ylabel('V [l]')
plt.title("Volume [l]")

plt.subplot(3, 2, 2)
plt.plot(res_industrial[:, 1], color='blue')
plt.autoscale(enable=True, axis='both', tight=True)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.xlabel('Time [h]')
plt.ylabel('cX [g/l]')
plt.title("Yeast concentration [g/l]")

plt.subplot(3, 2, 3)
plt.plot(res_industrial[:, 2], color='blue')
plt.autoscale(enable=True, axis='both', tight=True)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.xlabel('Time [h]')
plt.ylabel('cP [g/l]')
plt.title("Ethanol concentration [g/l]")

plt.subplot(3, 2, 4)
plt.plot(res_industrial[:, 3], color='blue')
plt.autoscale(enable=True, axis='both', tight=True)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.xlabel('Time [h]')
plt.ylabel('cS [g/l]')
plt.title("Glucose concentration [g/l]")

plt.subplot(3, 2, 5)
plt.plot(res_industrial[:, 4], color='blue')
plt.autoscale(enable=True, axis='both', tight=True)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.xlabel('Time [h]')
plt.ylabel('cO2 [g/l]')
plt.title("Oxygen concentration [g/l]")

plt.subplot(3, 2, 6)
plt.plot(res_industrial[:, 5], color='blue')
plt.autoscale(enable=True, axis='both', tight=True)
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.xlabel('Time [h]')
plt.ylabel('T [\u00B0C]')
plt.title("Reactor temperature [\u00B0C]")

plt.show()
