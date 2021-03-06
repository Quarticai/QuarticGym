.. _QuarticGym: https://github.com/Quarticai/QuarticGym

.. _Openai Gym: https://gym.openai.com/

.. _d4rl: https://github.com/rail-berkeley/d4rl.git

.. _PenSimPy: https://github.com/Mohan-Zhang-u/PenSimPy.git

.. _fastodeint: https://github.com/Quarticai/fastodeint.git

The `QuarticGym`_ supplements several process control environments to the `Openai Gym`_ family, which quenches the pain of performing Deep Reinforcement Learning algorithms on them. Furthermore, we provided `d4rl`_-like wrappers for accompanied datasets, making Offline RL on those environments even smoother.

Install
-------
.. code-block::

    $ git clone --recurse-submodules git@github.com:Quarticai/QuarticGym.git
    $ cd QuarticGym
    $ pip install .

.. note::
    You will need to build the `PenSimPy`_ environment with `QuarticGym`_ separately. Namely, build and install `fastodeint`_ following `this instruction <https://github.com/Quarticai/fastodeint/blob/master/README.md>`_, then install `PenSimPy`_.

    For Linux users, you can just install `fastodeint`_ and `PenSimPy`_ by executing the following commands:

    .. code-block::

        $ sudo apt-get install libomp-dev
        $ sudo apt-get install libboost-all-dev
        $ git clone --recursive git@github.com:Mohan-Zhang-u/fastodeint.git
        $ cd fastodeint
        $ pip install .
        $ cd ..
        $ git clone --recursive git@github.com:Mohan-Zhang-u/PenSimPy.git
        $ cd PenSimPy
        $ pip install .

    If you also want to use the pre-built MPC and EMPC controllers, you would need to install mpctools by CasADi. For Linux users, you can execute the following commands:

    .. code-block::

        $ git clone --recursive git@github.com:Mohan-Zhang-u/mpc-tools-casadi.git
        $ cd mpc-tools-casadi
        $ python mpctoolssetup.py install --user


Example Usage
-------------

See the `jupyter notebook <https://github.com/Quarticai/QuarticGym/blob/main/examples.ipynb>`_ for example use cases.
