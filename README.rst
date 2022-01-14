.. _QuarticGym: https://github.com/Quarticai/QuarticGym

.. _Openai Gym: https://gym.openai.com/

.. _d4rl: https://github.com/rail-berkeley/d4rl.git

.. _PenSimPy: https://github.com/Mohan-Zhang-u/PenSimPy.git

.. _fastodeint: https://github.com/Quarticai/fastodeint.git

The `QuarticGym`_ supplements several process control environments to the `Openai Gym`_ family, which quenches the pain of performing Deep Reinforcement Learning algorithms on them. Furthermore, we provided `d4rl`_-like wrappers for accompanied datasets, make Offline RL on those environments even smoother.

Install
-------
.. code-block::

    $ git clone --recurse-submodules git@github.com:Quarticai/QuarticGym.git
    $ cd QuarticGym
    $ pip install .

.. note::
    if you want to use the `PenSimPy`_ environment with `QuarticGym`_, you will have to build and install `fastodeint`_ following `this instruction <https://github.com/Quarticai/fastodeint/blob/master/README.md>`_, then install `PenSimPy`_.

    For Linux users, you can just install `fastodeint`_ and `PenSimPy`_ by executing the following commands:

    .. code-block::

        $ sudo apt-get install libomp-dev
        $ sudo apt-get install libboost-all-dev
        $ git clone --recursive git@github.com:Mohan-Zhang-u/fastodeint.git
        $ cd fastodeint
        $ pip install .
        $ cd ..
        $ git clone --recursive git@github.com:Mohan-Zhang-u/PenSimPy.git
        $ cd PenSimpy
        $ pip install .

Example Usage
-------------

You may want to consult this `jupyter notebook <https://github.com/Quarticai/QuarticGym/blob/main/examples.ipynb>`_ to see some example use cases.
