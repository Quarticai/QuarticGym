![alt text](figures/logo_light.png "Logo Title Text 1")

**Development Status:** As of 07/2021 PenSimPy is under active maintenance (expect bug fixes and updates). 
Please follow, star, and fork to get the latest functions.
# QuarticGym ![](https://img.shields.io/badge/python-3.8.0-orange) ![](https://img.shields.io/badge/c%2B%2B-11-blue)

TODO

Installation
============
The general requirements are listed in the [requirements.txt](requirements.txt)

### PenSimPy
PenSimPy relies on [fastodeint](https://github.com/Quarticai/fastodeint), a python bindings for an ODE solver implemented in C++. Unfortunely, it's not available as a PyPI package at this point, so you need to follow the installation instruction [here](https://github.com/Quarticai/fastodeint/blob/master/README.md) to build it yourself.
Once fastodeint is installed, you can install PenSimPy by the following command
```
pip install pensimpy
```
Examples
============
Please consult [our jupyter notebook](examples.ipynb)

