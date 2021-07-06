![alt text](figures/logo_light.png "Logo Title Text 1")

**Development Status:** As of 07/2021 PenSimPy is under active maintenance (expect bug fixes and updates). 
Please follow, star, and fork to get the latest functions.
# QuarticGym ![](https://img.shields.io/badge/python-3.8.0-orange) ![](https://img.shields.io/badge/c%2B%2B-11-blue)

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

Examples
============
Please consult [our jupyter notebook](examples.ipynb)

