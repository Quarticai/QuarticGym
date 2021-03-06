AtropineEnv
===========

Introduction
------------

What is atropine? We can simply treat it as one medication that benefits human beings. A corresponding continuous production flow can be described as below:

.. image:: imgs/Atropine_Manufacturing_Process.png
   :alt: Fig. 1

The whole flow contains *six* stream inputs, *two* outputs (product and
waste stream), *three* mixers, *three* tubular reactors, and *one*
liquid-liquid separator. They are connected sequentially and the flow
directions can be shown as those arrows. Besides, in our project, the
flow rates of NO. 5 and 6 are kept constant.

Atropine Model
--------------

Here comes the 1st challenge, how to model the whole flow? Since the
flow has duplicate components, like input streams, mixers, and tubular
reactors, we can develop single
`Stream <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L8>`__,
`Mixer <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L66>`__,
`TubularReactor <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L91>`__,
`LLSeparator <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L158>`__,
and
`Plant <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L214>`__
python classes and reuse them. They are based on this
`paper <https://ieeexplore.ieee.org/document/9147331>`__. Let us go
through each of them.

-  `Stream <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L8>`__

For each input stream, it has 14 reactants and they can be found
`here <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/helpers/constants.py#L10>`__.
This model is simple and no differential equations are included. It just
helps calculate the related *mass flow rates*, *molar concentrations*,
and *mass concentrations* based on the input *volumetric flow rates*.

-  `Mixer <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L66>`__

It is supposed that all the reactants are fully and intensively mixed
and no reaction happened (no new species generated). By following a
basic mass conservation equation, the
`model <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L80>`__
can be built.

-  `TubularReactor <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L91>`__

This model becomes complex because the partial differential equations
(PDEs) are included. The partial derivative of the *molar concentration*
of each species w.r.t *time* is
`modeled <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L144>`__.

.. image:: imgs/eq1.png

where :math:`c_{i,l}` is the molar concentration of species *i* at
discretization point *l*; *V* is the reactor volume; :math:`Q_{tot}`
is the total volumetric flow rate; :math:`r_{i,l}` is the reaction rate matrix.
This model is time-consuming because it has a double
`for-loop <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L125>`__
that not only iterates all discretized points (:math:`n_{d}` =40) but also
iterates all 14 species.

-  `LLSeparator <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L160>`__

This part is also based on the mass conservation equation but the
differential-algebraic equations (DAEs) are introduced.

.. image:: imgs/eq2.png

where :math:`F_{OR,i}` and :math:`F_{AQ,i}` are the molar flow rates of species *i* at
the organic and aqueous phases, respectively. Also, :math:`F_{OR,i}` and
:math:`F_{AQ,i}` are algebraic variables. Hence, the
`derivatives <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L173>`__
can be calculated based on the above equation, so as the
`algebraic <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L186>`__.

-  `Plant <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L214>`__

Now, let us assemble the above components together and get the whole
continuous flow. They are connected like
`this <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L216>`__
and are exactly following the design in Fig. 1. The `updated
states <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L319>`__
are based on the
`derivatives <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L453>`__
of the three tubular reactors and one liquid-liquid separator, and the
`algebraic <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L493>`__
of the liquid-liquid separator. Additionally, we have an extra step that
takes the `mixing
process <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L326>`__
into consideration right before calculating the `updated
state <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/models/atropine_process.py#L319>`__
by `CASADI <https://web.casadi.org/>`__. In a nutshell, the **Stream**
and **Mixer** help us preprocess the data, and the **TubularReactor**
and **LLSeparator** provide the derivative and the algebraic for
updating states.

Control System
--------------

System Design
~~~~~~~~~~~~~

So far, we have a model that represents the whole flow and it is an
open-loop one. The next step is to design a linear MPC-based control
system to track the steady-state inputs and output ASAP. First of all,
let us design the overall control system as below:

.. image:: imgs/Control_System.png
   :alt: Fig. 2 Control System

where the **Process** refers to our plant model and **State Estimator**
refers to a `Kalman
filter <https://en.wikipedia.org/wiki/Kalman_filter>`__ that is used for
states estimation; :math:`r(t)` is the reference signal; :math:`u(t)` is the inputs
for the plant model and the Kalman filter; :math:`y(t)` is the output; :math:`x(t)`
is the states and :math:`\hat{x}(t)` is the estimation. The above linear system
can be described below:

.. code::

   x(k+1) = Ax(k) + Bu(k)

   y(k) = Cx(k) + Du(k)

kalman filter progresses like:

.. code::

   e(k) = y(k) - yhat(k)

   xhat(k+1) = Axhat(k) + Bu(k) - Ke(K)

where *e* is the error.

System Identification
~~~~~~~~~~~~~~~~~~~~~

In order to get the above matrics **A**, **B**, **C**, **D,** and **K**,
we can take advantage of a system identification package
`SIPPY <https://github.com/CPCLAB-UNIPI/SIPPY>`__. And training data can be generated by the open-loop model for the identification process.

-  With respect to the `open-loop model <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/atropineenv.py#L140>`_, it has four inputs (volumetric
   flow rates u1, u2, u3, and u4 in Fig. 1) and one output (e-factor)
   that describes atropine production efficiency.

   .. image:: imgs/eq3.png

   The input data of the open-loop model can be acquired by random noise
   and the corresponding e-factor can be
   `calculated <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/atropineenv.py#L167>`__
   based on the algebraic (less than 0.1s per run). For our case, it
   seems only 200 rows of data is enough for linear model
   identification.

-  Similar to the ML model training, a training and testing data split
   method was applied.
   However, the linear trend of the output data requires to be removed and the processed output signal is shown below.

   .. image:: imgs/De-trendedOutputSignalvsTheOriginal.png
      :alt: Fig. 3 De-trended Output Signal vs The Original

-  System identification is pretty simple and we chose an
   identification method of N4SID and an order of
   2.
   Also, a `performance
   metric <https://github.com/Quarticai/QuarticGym/blob/fa72b15f63a73f42eb232a5ae12fa0e216183da1/quarticgym/envs/helpers/helper_funcs.py#L5>`__
   is designed. Finally, after the system identification, we have the
   following system:

.. math::

   x(k+1) = \begin{bmatrix} 0.8314 & -0.235 \\ 0.1032 & 0.8634 \end{bmatrix} x(k) + \begin{bmatrix} 0.2570 & 1.8423 & 0.2785 & -0.6279 \\ -0.3092 & -0.7418 & -0.1148 & 0.3814 \end{bmatrix} u(k) \\

.. math::

   y(k) = \begin{bmatrix} -8.6962 & -9.7070 \end{bmatrix} x(k) \\

.. math::

   K = \begin{bmatrix} -0.0279 \\ -0.0243 \end{bmatrix}

System Implementation
~~~~~~~~~~~~~~~~~~~~~

We generated below the trajectory with `MPC <https://en.wikipedia.org/wiki/Model_predictive_control>`_, and ended the simulation once it reaches the steady state (we considered this as a termination condition)
The whole simulation represents 400 minutes in real life and the results are shown below:

.. image:: imgs/mpc1.png
   :alt: Fig. 4 Track of E-factor

.. image:: imgs/mpc2.png
   :alt: Fig. 5 Track of Four Inputs

It can be seen that the control system successfully worked and both the
inputs and the output reach the steady-state. The whole process (data generation, system identification,
and MPC control) takes less than 10 seconds.

AtropineEnv module
------------------

.. automodule:: quarticgym.envs.atropineenv
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
