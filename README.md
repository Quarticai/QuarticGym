![alt text](figures/logo_light.png "Logo Title Text 1")

**Development Status:** As of 07/2021 this repo is under active maintenance. Please follow, star, and fork to get the latest functions.
# QuarticGym ![](https://img.shields.io/badge/python-3.8.0-orange)

The QuarticGym supplements several Process Control environments to the [Openai Gym](https://gym.openai.com/) family, which quenches the pain of performing Deep Reinforcement Learning algorithms on them. Furthermore, we provided [d4rl-like](https://github.com/rail-berkeley/d4rl.git) wrappers for accompanied datasets, make Offline RL on those environments even smoother.

Installation
============
The general requirements are listed in the [requirements.txt](requirements.txt)

### PenSimPy
[PenSimPy](https://github.com/Quarticai/PenSimPy) relies on [fastodeint](https://github.com/Quarticai/fastodeint), a python bindings for an ODE solver implemented in C++. Unfortunely, it's not available as a PyPI package at this point, so you need to follow the installation instruction [here](https://github.com/Quarticai/fastodeint/blob/master/README.md) to build it yourself.
Once fastodeint is installed, you can install PenSimPy by the following command
```
pip install pensimpy
```

### BeerFMT
BeerFMT simulates the Beer Fermentation process.

### AtropineEnv
AtropineEnv simulates an atropine production environment to accommodate the [Atropine_Challenge](https://github.com/Quarticai/Atropine-Challenge).

### ReactorEnv
ReactorEnv simulates a general reactor environment. This is supposed to be an template environment. The documentations in that file is enhanced and provided comment lines (# ---- standard ---- and # /---- standard ----) enclose pieces of code that should be reused by most of QuarticGym environments. I will extend some of them into a base class in the future.

Examples
============
Please consult [our jupyter notebook](examples.ipynb)
