======
MAbEnv
======

.. role:: raw-latex(raw)
   :format: latex
..

Introduction to Modeling and Operation of a Continuous Monoclonal Antibody (mAb) Manufacturing Process
------------------------------------------------------------------------------------------------------

Drugs based on monoclonal antibodies (mAbs) play an indispensable role
in biopharmaceutical industry in aspects of therapeutic and market
potentials. In therapy and diagnosis applications, mAbs are widely used
for the treatment of autoimmune diseases, and cancer, etc. According to
a recent publication, mAbs also show promising results in the treatment
of COVID-19 :raw-latex:`\cite{wang2020human}`. Until September 22, 2020,
94 therapeutic mAbs have been approved by U.S. Food & Drug
Administration (FDA) :raw-latex:`\cite{antibody2020antibody}` and the
number of mAbs approved within 2010-2020 is three times more than those
approved before 2010 :raw-latex:`\cite{kaplon2020antibodies}`. In terms
of its market value, it is expected to reach a value of $198.2 billion
in 2023. Thus, with the fact that Canada is an active and competitive
contributor to the development of high capacity mAb manufacturing
processes :raw-latex:`\cite{nserc2018nserc}`, increasing the production
capacity of mAb manufacturing processes is immediately necessary due to
the explosive growth in mAb market. Integrated continuous manufacturing
of mAbs represents the state-of-the-art in mAb manufacturing and has
attracted a lot of attention, because of the steady-state operations,
high volumetric productivity, and reduced equipment size and capital
cost, etc :raw-latex:`\cite{croughan2015future}`. However, there is no
existing mathematical model of the integrated manufacturing process and
there is no optimal control algorithm of the entire integrated process.
This project fills the knowledge gaps by first developing a mathematical
model of the integrated continuous manufacturing process of mAbs.
Working towards an optimal control algorithm for mAbs manufacturing, two
variants of advanced process control (APC) algorithms namely Model
Predictive Control (MPC) and Economic Model Predictive Control (EMPC)
are designed and tested on the mAb production process.

Mathematical model development
==============================

In this chapter, we present a physics-based mathematical model of the
Monoclonal Antibody (mAb) production process. The mAb production process
consists of two sub-processes referred to in this work as the upstream
process and the downstream processes. The upstream model presented here
is primarily based on the works by Kontoravdi et al.
:raw-latex:`\cite{KONTORAVDI2008,KONTORAVDI2010}` as well as other works
in literature and the downstream model is mainly based on the works by
Gomis-Fons et al. :raw-latex:`\cite{gomis2020model}`. We begin the
chapter by first describing the mAb production process. Subsequently, we
present the mathematical models of the various units in the mAb
production process.

Process description
-------------------

As mentioned earlier, the mAb production process consists of the
upstream and the downstream processes. In the upstream process, mAb is
produced in a bioreactor which provides a conducive environment mAb
growth. The downstream process on the other hand recovers the mAb from
the upstream process for storage. In the upstream process for mAb
production, fresh media is fed into the bioreactor where a conducive
environment is provided for the growth of mAb. A cooling jacket in which
a coolant flows is used to control the temperature of the reaction
mixture. The contents exiting the bioreactor is passed through a
microfiltration unit which recovers part of the fresh media in the
stream. The recovered fresh media is recycled back into the bioreactor
while the stream with high amount of mAb is sent to the downstream
process for further processing. A schematic diagram of upstream process
is shown in Figure `2.1 <#fig:upstream>`__.

.. container:: center

   .. figure:: imgs/upstream_process.png
      :alt: A schematic diagram of the upstream process for mAb
      production
      :name: fig:upstream
      :width: 90.0%

      A schematic diagram of the upstream process for mAb production

The objective of the downstream process for mAb production is to purify
the stream with high concentration of mAb from the upstream and obtain
the desired product. The configuration of the downstream is adopted from
Gomis-Fons’ work :raw-latex:`\cite{gomis2020model}`. It is composed of a
set of fractionating columns, for separating mAb from impurities, and
holdup loops, for virus inactivation (VI) and pH conditioning. The
schematic diagram of downstream process is shown in Figure
`2.2 <#fig:downstream>`__. Three main steps are considered in the
scheme, capture, virus inactivation, and polish. It worth mentioning
that the ultra filtration preparing the final product is not considered
in this work hence is not included in the diagram. The capture step
serves as the main component in the downstream and the majority of mAb
is recovered in this step. Protein A chromatography columns are usually
utilized to achieve this goal. The purpose of VI is to disable the virus
and prevent further mAb degradation. At last, the polish step further
removes the undesired components caused by VI and cation-exchange
chromatography (CEX) and anion-exchange chromatography (AEX) are
generally used. In order to obtain a continuous manufacturing process,
the perfusion cell culture, a continuous mAb culturing process is used
in the upstream, however, the nature of chromatography is discontinuous.
Therefore, a twin-column configuration is implemented in the capture
step. According to the diagram, the column A is connected to the stream
from the upstream and loaded with the solutions. Simultaneously, the
column B is connected to the remaining components of the downstream and
conducting the further mAb purification. According to Gomis-Fons, et al.
:raw-latex:`\cite{gomis2020model}`, the time needed for loading is
designed as the same as the time required for the remaining purification
steps. Hence, when the column A is fully loaded, the column B is empty
and resin inside is regenerated. Then, the roles of these two columns
will be switched in the new configuration, meaning the column B will be
connected to the upstream and column A will be connected to the
remaining components in the downstream. In conclusion, a continuous
scheme of downstream is achieved by implementing the twin-column
configuration in the capture step.

.. container:: center

   .. figure:: imgs/downstream_process.png
      :alt: A schematic diagram of the downstream process for mAb
      production
      :name: fig:downstream
      :width: 90.0%

      A schematic diagram of the downstream process for mAb production

Bioreactor modeling
-------------------

The mathematical model of the bioreactor can be divided into three parts
namely cell growth and death, cell metabolism, and mAb synthesis and
production. Papathanasiou and coworkers described a simplified metabolic
network of GN-NS0 cells using a Monod kinetic model
:raw-latex:`\cite{PAPATHANASIOU2017}`. In the study by Villiger et al.
:raw-latex:`\cite{VILLIGER2016}`, while the specific productivity of mAb
was observed to be constant with respect to viable cell density, it
varied with respect to the extracellular pH. By considering these two
models, we proposed one simplified model to describe the continuous
upstream process. The following assumptions were used in developing the
dynamic model of the bioreactor in the continuous production of mAb
process.

-  The contents of the bioreactor is perfectly mixed

-  The dilution effect is negligible

-  The enthalpy change due to cell death is negligible

-  There is no heat loss to the external environment

-  The temperature of the recycle stream and the temperature of the
   reaction mixture are equal

Cell growth and death
~~~~~~~~~~~~~~~~~~~~~

An overall material balance on the bioreactor yields the equation

.. math::
   :name: eq:mb:vessel

   \begin{aligned}
    \label{eq:mb:vessel}
       \frac{d{V_1}}{dt} &= F_{in} + F_{r} - F_{out} \end{aligned}

Try link :ref:`Link title <eq:mb:vessel>`, In Equation `[eq:mb:vessel] <#eq:mb:vessel>`__, :math:`V` is the volume
in :math:`L`, and :math:`F_{in}`, :math:`F_{r}`, and :math:`F_{out}` are
the volumetric flow rate of the fresh media into the reactor, the
volumetric flow rate of the recycle stream and the volumetric flow rate
out of the bioreactor respectively in :math:`L/h`. Throughout this
report, the subscripts :math:`1` and :math:`2` represent the bioreactor
and the microfiltration unit respectively.

The conversion of the viable and total cells within the culture can be
described using a component balance on the viable and total number of
cells as shown in
Equations `[eq:mb:viablecell:R1] <#eq:mb:viablecell:R1>`__ and
`[eq:mb:totalcell:R1] <#eq:mb:totalcell:R1>`__

.. math::

   \begin{aligned}
       \frac{dX_{v1}}{dt} &= \mu X_{v1} - \mu_d X_{v1} - \frac{F_{in}}{V_1}X_{v1}+ \frac{F_{r}}{V_1}(X_{vr}-X_{v1}) \label{eq:mb:viablecell:R1}\\
       \frac{dX_{t1}}{dt} &=  \mu X_{v1} -  \frac{F_{in}}{V_1}X_{t1} + \frac{F_{r}}{V_1}(X_{tr}-X_{t1}) \label{eq:mb:totalcell:R1}\end{aligned}

where :math:`X` is the cell concentration in :math:`cells/L`,
:math:`\mu` is the specific growth rate in :math:`h^{-1}` and
:math:`\mu_d` is the specific death rate in :math:`h^{-1}`. The
subscripts :math:`v` and :math:`t` denote the viable and total cells
respectively.

The specific cell growth rate is determined by the concentrations of the
two key nutrients namely glucose and glutamine, the two main metabolites
namely lactate and ammonia and temperature following the Monod kinetics

.. math::

   \begin{aligned}
       \mu &= \mu_{max}f_{lim}f_{inh} \label{eq:mu}\\
       f_{lim} &= (\frac{[GLC]_1}{K_{glc}+[GLC]_1})(\frac{[GLN]_1}{K_{gln}+[GLN]_1}) \label{eq:flim} \\
       f_{inh} &= (\frac{KI_{lac}}{KI_{lac}+[LAC]_1}) (\frac{KI_{amn}}{KI_{amn}+[AMN]_1}) \label{eq:finh}\end{aligned}

In Equation `[eq:mu] <#eq:mu>`__, :math:`\mu_{max}` is the maximum
specific growth rate in :math:`h^{-1}`, :math:`f_{lim}` and
:math:`f_{inh}` are the nutrient limitation function and the product
inhibition function which are described in
Equations `[eq:flim] <#eq:flim>`__ and `[eq:finh] <#eq:finh>`__,
respectively. In Equations `[eq:flim] <#eq:flim>`__ and
`[eq:finh] <#eq:finh>`__, :math:`[GLC]`, :math:`[GLN]`, :math:`[LAC]`
and :math:`[AMM]` stands for the concentrations of glucose, glutamine,
lactate and ammonia in :math:`mM`, and :math:`K_{glc}`, :math:`K_{gln}`,
:math:`KI_{lac}` and :math:`KI_{amm}` represent the Monod constant for
glucose, glutamine, lactate and ammonia respectively in :math:`mM`.

The specific death rate is determined based on the assumption that cell
death is only a function of the concentration of ammonia accumulating in
the culture, and is shown as follow:

.. math::

   \label{eq:mu_d}
       \mu_d = \frac{\mu_{d,max}}{1+(\frac{K_{d,amm}}{[AMM]_1})^n}; ~ n > 1

In Equation `[eq:mu_d] <#eq:mu_d>`__, :math:`n` is assumed to be greater
than 1 to give a steeper increase of specific death as ammonia
concentration increases.

Temperature is a key factor in the maintenance of cell viability and
productivity in bioreactors. It is expected that the growth and death of
the mAb-producing cells will be affected by temperature. The effect of
temperature on the specific growth and death rates is achieved through
the maximum specific growth and death rates. In this study, standard
linear regression of data available in literature
:raw-latex:`\cite{jimenez2016}` was used to obtain a linear relationship
between the temperature and the maximum cell growth rate
:math:`\mu_{\text{max}}`.

.. math::

   \label{eqn:max_specific_growth}
       \mu_{\text{max}} = 0.0016T - 0.0308

Similarly, a linear relationship was obtained for the maximum cell death
rate as shown in

.. math::

   \label{eqn:max_specific_death}
       \mu_{d,\text{max}} = -0.0045T + 0.1682

In `[eqn:max_specific_growth] <#eqn:max_specific_growth>`__ and
`[eqn:max_specific_death] <#eqn:max_specific_death>`__, :math:`T` is the
temperature of the bioreactor mixture in :math:`^\circ C`. The data was
obtained for the maximum specific growth and death rates at 33
:math:`^{\circ}`\ C and 37 :math:`^{\circ}`\ C. Therefore, the Equations
`[eqn:max_specific_growth] <#eqn:max_specific_growth>`__ and
`[eqn:max_specific_death] <#eqn:max_specific_death>`__ are valid only
within this temperature range. A heat balance on the bioreactor together
with the following above assumptions leads to the following ordinary
differential equation:

.. math::

   \label{eqn:temp}
       \frac{dT}{dt}=\frac{F_{in}}{V_1}(T_{in}-T) +\frac{-\Delta H}{\rho c_p}(\mu X_{v1}) + \frac{U}{V_1 \rho  c_p}(T_c - T)

In Equation `[eqn:temp] <#eqn:temp>`__, :math:`T_{in}` is the
temperature of the fresh media in :math:`^{\circ} C`, :math:`\Delta H`
is the heat of reaction due to cell growth in :math:`J/mol`,
:math:`\rho` is the density of the reaction mixture in :math:`g/L`,
:math:`c_p` is the specific heat capacity of the reaction in
:math:`J/(g \circ C)`, :math:`U` is the overall heat transfer
coefficient in :math:`J/(hr ^\circ C)`), and :math:`T_c` is the
temperature of fluid in the jacket in :math:`^{\circ}`\ C.

The first term of Equation `[eqn:temp] <#eqn:temp>`__ represents the
heat transfer due to the inflow of the feed and the second term
represents the heat consumption due to the growth of the cells. The
final term describes the external heat transfer to the bioreactor due to
the cooling jacket.

Cell metabolism
~~~~~~~~~~~~~~~

A mass balance on glucose, glutamine, lactate and ammonia around the
bioreactor results in the following equations
:raw-latex:`\cite{PAPATHANASIOU2017}`:

.. math::

   \begin{aligned}
       \frac{d[GLC]_1}{dt} & = -Q_{glc}X_{v1} +  \frac{F_{in}}{V_1} ([GLC]_{in} - [GLC]_1) + \frac{F_{r}}{V_1}([GLC]_r-[GLC]_1) \\
       Q_{glc} &= \frac{\mu}{Y_{X,glc}} + m_{glc} \\
       \frac{d[GLN]_1}{dt} &= - Q_{gln}X_{v1} - K_{d,gln}[GLN]_1 +  \frac{F_{in}}{V_1}([GLN]_{in} - [GLN]_1) -  \frac{F_{r}}{V_1}([GLC]_1 - [GLN]_1)\\
       Q_{gln} &= \frac{\mu}{Y_{X,gln}} + m_{gln} \\
       m_{gln} &= \frac{\alpha_1 [GLN]_1}{\alpha_2+[GLN]_1}\end{aligned}

.. math::

   \begin{aligned}
       \frac{d[LAC]_1}{dt} &= Q_{lac}X_{v1} -  \frac{F_{in}}{V_1}[LAC]_1 + \frac{F_r}{V_1}([LAC]_r-[LAC]_1) \\
       Q_{lac} &= Y_{lac,glc}Q_{glc}\\
       \frac{d[AMM]_1}{dt} &= Q_{amm}X_{v1} + K_{d,gln}[GLN]_1 -  \frac{F_{in}}{V_1}[AMM]_1 + \frac{F_r}{V_1}([AMM]_r-[AMM]_1) \\
       Q_{amm} &= Y_{amm,gln}Q_{gln}\end{aligned}

MAb production
~~~~~~~~~~~~~~

The rate of mAb production is described as

.. math::

   \begin{aligned}
       \frac{d[mAb]_1}{dt} &= X_{v1} Q_{mAb} -  \frac{F_{in}}{V_1}[mAb]_1 + \frac{F_r}{V_1}([mAb]_r-[mAb]_1) \label{eq:mAb:R1} \\
       Q_{mAb} &= Q_{mAb}^{max} exp[-\frac{1}{2}(\frac{pH-pH_{opt}}{\omega_{mAb}})^2] \label{eq:QmAb:R1}\end{aligned}

In Equation `[eq:QmAb:R1] <#eq:QmAb:R1>`__, :math:`Q_{mAb}^{max}` is the
maximum specific productivity with unit :math:`mg/cell/h`, and
:math:`\omega_{mAb}` is the pH-dependent productivity constant.
:math:`pH_{opt}` is the optimal culture pH as shown in
:raw-latex:`\cite{VILLIGER2016}`. The pH value is assumed as a function
of state and shown in Section `2.3.2 <#sec:pH>`__.

Mircofiltration
---------------

Cell separation
~~~~~~~~~~~~~~~

In the cell separation process, a external hollow fiber (HF) filter is
used as cell separation device. It is assumed that no reactions occur in
the separation process. Hence, the concentration of each variable in
recycle stream is shown as follows:

.. math::

   \begin{aligned}
       X_{vr} & = \eta_{rec} X_{v1}\frac{F_1}{F_r}\\
       X_{tr} & = \eta_{rec} X_{t1}\frac{F_1}{F_r}\\
       [GLC]_r & = \eta_{ret} [GLC]_1\frac{F_1}{F_r}\\
       [GLN]_r & = \eta_{ret} [GLN]_1\frac{F_1}{F_r}\\
       [LAC]_r & = \eta_{ret} [LAC]_1\frac{F_1}{F_r}\\
       [AMM]_r & = \eta_{ret} [AMM]_1\frac{F_1}{F_r}\\
       [mAb]_r & = \eta_{ret} [mAb]_1\frac{F_1}{F_r}\end{aligned}

According to :raw-latex:`\cite{clincke2013}`, the cell recycle rate
(:math:`\eta_{rec}`) is assumed to be 92\ :math:`\%` and the retention
rates of glucose, glutamine, lactate, ammonia, and mAb
(:math:`\eta_{ret}`) are assumed to be 20\ :math:`\%`.

The material balance around the separation device is shown as:

.. math:: \frac{dV_2}{dt} = F_1  - F_2 -F_r

The mass balance for concentrations of glucose, glutamine, lactate,
ammonia, and mAb can be described as:

.. math::

   \begin{aligned}
       \frac{dX_{v2}}{dt} &= \frac{F_1}{V_2}(X_{v1}-X_{v2}) - \frac{F_r}{V_2} (X_{vr} - X_{v2}) \\
       \frac{dX_{t2}}{dt} &= \frac{F_1}{V_2}(X_{t1}-X_{t2}) - \frac{F_r}{V_2} (X_{tr} - X_{t2}) \\
       \frac{d[GLC]_2}{dt} &= \frac{F_1}{V_2}([GLC]_1-[GLC]_2) - \frac{F_r}{V_2} ([GLC]_r - [GLC]_2) \\
       \frac{d[GLN]_2}{dt} &= \frac{F_1}{V_2}([GLN]_1-[GLN]_2) - \frac{F_r}{V_2} ([GLN]_r - [GLN]_2) \\
       \frac{d[LAC]_2}{dt} &= \frac{F_1}{V_2}([LAC]_1-[LAC]_2) - \frac{F_r}{V_2} ([LAC]_r - [LAC]_2) \\
       \frac{d[AMM]_2}{dt} &= \frac{F_1}{V_2}([AMM]_1-[AMM]_2) - \frac{F_r}{V_2} ([AMM]_r - [AMM]_2) \\
       \frac{d[mAb]_2}{dt} &= \frac{F_1}{V_2}([mAb]_1-[mAb]_2) - \frac{F_r}{V_2} ([mAb]_r - [mAb]_2) \end{aligned}

.. _sec:pH:

pH value
~~~~~~~~

pH is defined as the decimal logarithm of the reciprocal of the hydrogen
ion activity in a solution. We assume our pH model as follows:

.. math:: pH = \theta_1 - log_{10}(\theta_2 [AMM] +\theta_3)

After applying nonlinear regression method, we fit the model as:

.. math:: pH = 7.1697 - log_{10}(0.074028[AMM] +0.968385)

Downstream modeling
-------------------

The mathematical model of the downstream is constructed based on each
unit operation. Specifically, two different models are utilized to
describe the loading mode and elution mode of the Protein A
chromatography column, separately. The models for CEX and AEX share the
same mathematical structure with different parameters and the models for
VI and holdup loop share the same structures and parameters. The
detailed explanation of each model is shown in the following
subsections.

Protein A chromatography column loading mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A schematic diagram :raw-latex:`\cite{dizaji2016minor}` depicting a
general chromatography column is shown in
Figure `2.3 <#fig:chromatography>`__. The column is packed with the
porous media which have the binding sites with mAb. The porous media is
defined as the stationary phase and the fluid which contains mAb and
flows through the column is considered as the mobile phase. Three types
of mass transfers are usually considered inside of the column. From the
top of the figure, the convection caused by the bulk fluid movement is
portrayed. Then by only considering a control volume of the column,
which is illustrated in the second subfigure, the dispersion of mAb
along the axial direction is shown. Within the beads, there is
intra-particle diffusion and at the last subfigure, mAbs are adsorbed on
the binding sites of beads.

.. container:: center

   .. figure:: imgs/chromatography.png
      :alt: A schematic diagram of the chromatography column
      :name: fig:chromatography
      :width: 60.0%

      A schematic diagram of the chromatography column

The general rate model (GRM) simulates the mass transfer in
chromatography column, with the assumption that the transfer along the
radial direction of the column is negligible and the transfer along the
axial direction of the column and the radial direction in the beads are
considered.

In this work, the GRM identified by Perez-Almodovar and Carta
:raw-latex:`\cite{perez2009igg}` is used to describe the loading mode of
the Protein A chromatography column. The mass transfer along the axial
coordinate is described below:

.. math::

   \label{eq:capture:mobilephase}
       \frac{\partial c}{\partial t} = D_{ax} \frac{\partial^2 c}{\partial z^2} - \frac{v}{\epsilon_c}\frac{\partial c}{\partial z} - \frac{1-\epsilon_c}{\epsilon_c} \frac{3}{r_p} k_f(c-c_p |_{r=r_p})

where :math:`c` denotes the mAb concentration in the mobile phase,
changing with time (:math:`t`) and along the axial coordinates of
columns (:math:`z`). :math:`D_{ax}` is the axial dispersion coefficient,
:math:`v` is the superficial fluid velocity, :math:`\epsilon_c` is the
extra-particle column void, :math:`r_p` is the radius of particles and
:math:`k_f` is the mass transfer coefficient.

On the right hand side of
Equation `[eq:capture:mobilephase] <#eq:capture:mobilephase>`__, there
are three terms. The first term,
:math:`\frac{\partial^2 c}{\partial z^2}`, models the dispersion of mAb.
In other words, it describes the movement of mAb caused by the
concentration difference in the column. The second term
:math:`\frac{\partial c}{\partial z}` denotes the change of
concentration of mAb caused by the convection flow. The last term
:math:`k_f(c-c_p |_{r=r_p})` describes the mass transfer between the
mobile phase :math:`c` and the surface of the beads
:math:`c_p |_{r=r_p}`.

The boundary conditions of
Equation `[eq:capture:mobilephase] <#eq:capture:mobilephase>`__ are
shown below:

.. container:: subequations

   .. math::

      \begin{aligned}
          \frac{\partial c}{\partial z} &= \frac{v}{\epsilon_c D_{ax}} (c-c_F) \mbox{ ~~at~~} z=0 \label{eq:capture:mobilephase:boundary1}\\
          \frac{\partial c}{\partial z} &= 0 \mbox{ ~~at~~} z=L \label{eq:capture:mobilephase:boundary2}\end{aligned}

where :math:`c_F` stands for the harvest mAb concentration from upstream
process.

The concentration of mAb along radial coordinate in the beads
(:math:`c_p`) is the second component of GRM and the mass balance for
protein diffusion inside the porous particles is shown in
Equation `[eq:capture:particle] <#eq:capture:particle>`__ with boundary
conditions in
Equations `[eq:capture:particle:boundary1] <#eq:capture:particle:boundary1>`__
and `[eq:capture:particle:boundary2] <#eq:capture:particle:boundary2>`__

.. math::

   \label{eq:capture:particle}
        \frac{\partial c_p}{\partial t} = D_{eff} \frac{1}{r^2} \frac{\partial}{\partial r}(r^2 \frac{\partial c_p}{\partial r}) - \frac{1}{\epsilon_p} \frac{\partial (q_1 + q_2)}{\partial t}

.. container:: subequations

   .. math::

      \begin{aligned}
          \frac{\partial c_p}{\partial r} &= 0 \mbox{ ~~at~~} r=0 \label{eq:capture:particle:boundary1}\\
          \frac{\partial c_p}{\partial r} &= \frac{k_f}{D_{eff}} (c-cp) \mbox{ ~~at~~} r=r_p \label{eq:capture:particle:boundary2}\end{aligned}

where :math:`D_{eff}` is the effective pore diffusivity, :math:`r` is
the distance from the current location to the center of the particle,
and :math:`\epsilon_p` is the particle porosity.

At last, the description of adsorbed mAb concentration (:math:`q_1` and
:math:`q_2`) is shown as follows:

.. math::

   \label{eq:capture:adsobed}
      \frac{\partial q_i}{\partial t} = k_i [(q_{max,i}-q_i)c_p|_{r=r_p} - \frac{q_i}{K}] \mbox{ ~~for~~} i=1,2

where :math:`k_i` is the adsorption kinetic constant, :math:`q_{max}` is
the column capacity, and :math:`K` is the Langmuir equilibrium constant.
The reason of having two :math:`\frac{\partial q}{\partial t}` is
because there are two adsorption sites on the beads and one of them is
fast binding site and another one is the slow one.

Protein A chromatography column elution mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An adsorption kinetic model, convective-dispersive equation with
adsorption, is used to describe the elution of Protein A chromatography
column. The setup of boundary conditions for this model can take
Equations `[eq:capture:mobilephase:boundary1] <#eq:capture:mobilephase:boundary1>`__
and
`[eq:capture:mobilephase:boundary2] <#eq:capture:mobilephase:boundary2>`__
as the reference, at the same time keeping the inlet and outlet
conditions of elution mode in mind. The model is shown as follows:

.. math::

   \begin{aligned}
       \frac{\partial c}{\partial t} &= D_{ax} \frac{\partial^2 c}{\partial z^2} - \frac{v}{\epsilon}\frac{\partial c}{\partial z} + \frac{1-\epsilon_c}{\epsilon} \frac{\partial q}{\partial t} \label{eq:elution:mobilephase} \\
       \frac{\partial q}{\partial t} &= k [H_0 c_s^{-\beta} (1- \frac{q}{q_{max}})c-q] \label{eq:elution:adsorbed}\\
       \frac{\partial c_s}{\partial t} &=D_{ax}\frac{\partial^2c_s}{\partial z^2} - \frac{v}{\epsilon} \frac{\partial c_s}{\partial z} \label{eq:elution:modifier}\end{aligned}

where :math:`c` is the mAb concentration in the mobile phase,
:math:`c_s` stands for the modifier concentration, :math:`q` is the
adsorbed mAb concentration. :math:`k` is the adsorption/desorption rate,
:math:`H_0` is the Henry equilibrium constant, :math:`\beta` is the
equilibrium modifier-dependence parameter, :math:`\epsilon` is the total
column void.

On the right hand side of
Equation `[eq:elution:mobilephase] <#eq:elution:mobilephase>`__, the
first two terms are similar with those in
Equation `[eq:capture:mobilephase] <#eq:capture:mobilephase>`__. The
third term :math:`\frac{\partial q}{\partial t}` is detailed expressed
in Equation `[eq:elution:adsorbed] <#eq:elution:adsorbed>`__, which is a
Langmuir isotherm describing the adsorption and desorption of mAb on
beads. This mass transfer is affected by the concentration of the
modifier :math:`c_s` whose dynamics is described in
Equation `[eq:elution:modifier] <#eq:elution:modifier>`__.

CEX and AEX chromatography
~~~~~~~~~~~~~~~~~~~~~~~~~~

The adsorption kinetic model shown in
Equations `[eq:elution:mobilephase] <#eq:elution:mobilephase>`__, `[eq:elution:adsorbed] <#eq:elution:adsorbed>`__
and Equation `[eq:elution:modifier] <#eq:elution:modifier>`__ can also
be used to describe the CEX and AEX chromatography process. The same
rule applys to the boundary conditions. Since the AEX column is in
flow-through mode as described in :raw-latex:`\cite{perez2009igg}`, the
product mAb is not adsorbed on the beads and the kinetic constant
:math:`k` is supposed to be zero.

Virus inactivation and holdup pool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Equation `[eq:loops] <#eq:loops>`__ shows the model of loop for VI and
holdup, which is modeled as a one-dimensional dispersive-convective
transport, with boundary conditions in
Equations `[eq:capture:mobilephase:boundary1] <#eq:capture:mobilephase:boundary1>`__
and `[eq:capture:mobilephase:boundary2] <#eq:capture:mobilephase:boundary2>`__.
Since the loop is not packed, there is not intra-particle diffusion or
mass transfer between mAb outside of particles and on the surface of the
particles.

.. math::

   \label{eq:loops}
       \frac{\partial c}{\partial t} = D_{ax} \frac{\partial^2 c}{\partial z^2} - v\frac{\partial c}{\partial z}

Operation
=========

This chapter serves as a **README** of the digital twin of the
integrated continuous manufacturing process of mAb. The digital twin is
written in Python language. This Python project includes one file and
two folders. The file named **run.py** is used for executing the
project. The folder named **model** includes a python file named
**model_helper.py** that includes the code of digital twin. The second
folder named **utils** includes a python file named **utils_helper.py**
that includes the functions for saving and visualizing data.

Python file description
-----------------------

The three Python files will be discussed one by one in the remaining of
the subsection.

model_helper.py
~~~~~~~~~~~~~~~

In this file, two Python classes are defined named **UpModelHelper()**
and **DownModelHelper()**. The main idea is that each method in the
class is a digital twin of one unit operation. For example, the digital
twin of the upstream is named as **reactor()** under
**UpModelHelper()**, the digital twin of the loading mode of the capture
column is named as **capture_load()** under **DownModelHelper()**, etc.
Hence, in the initiation of the class, **\__init__()**, the number of
states for each digital twin, the number of inputs for each digital
twin, and/or the volume and the length of each unit operation are
initialized. Because the properties are specific for each unit
operation, the parameters for each digital twin are specified under each
corresponding method.

utils_helper.py
~~~~~~~~~~~~~~~

In this file, there is one class named **UtilsHelper** including several
methods, such as reading data from files, save results to a file, and
visualize trajectories of the digital twin.

run.py
~~~~~~

This file serves as the main file for the project. In this file, the
spatial and temporal parameters can be defined by the user, as well as
the initial condition of the system. Currently, we assume that the
downstream is continuously running only after the upstream reaches its
steady state. Hence, the flow rate and concentration of the stream from
upstream are constant. In addition, the downstream model should have run
repetitively. However, for each cycle, because the downstream will start
at the same initial conditions with the same inputs, only one cycle of
downstream is represented. At last, the code will utilize the methods in
**utils_helper.py** for data saving and visualization. The selected
state and input trajectories is shown in the following section with a
brief discussion.

Results and discussion
----------------------

Because of the nature of the first principle model, the numerical model
of digital twin of downstream is more stiff than that of upstream.
Therefore, a smaller time step (with the unit of minute) is required to
solve the ordinary differential equations of downstream, whereas the
upstream has the time step in the unit of hour. In the following
subsections, the selected state trajectories are shown to help the
understanding of the output the digital twins.

Upstream
~~~~~~~~

The trajectories of concentration of mAb in the reactor and the
separator are shown in Figure `3.1 <#fig:up_state>`__. According to the
trajectories, the system is able to reach a steady state under the
condition that the input is constant. Other state trajectories are also
able to reach their steady states.

.. container:: center

   .. figure:: imgs/up_state.png
      :alt: State trajectories of concentration of mAb in upstream
      :name: fig:up_state
      :width: 60.0%

      State trajectories of concentration of mAb in upstream

Downstream
~~~~~~~~~~

The trajectories of states in the capture column loading mode and the
elution mode are selected to show in the following
Figures `3.2 <#fig:down_state1>`__ and `3.3 <#fig:down_state2>`__,
respectively. In Figure `3.2 <#fig:down_state1>`__, the top three
subfigures shows the concentration of mAb at different locations inside
of the particles. By comparing the three subfigures, the trajectories
are quit similar. It shows that the concentration of mAb throughout the
radius of particles are similar. Now by checking the third subfigure
only, it reveals that the particles at the beginning of the column
(level 1) will encounter the inlet flow earlier than those at the end of
the column (level 49), furthermore, the concentration of mAb inside of
them will increase earlier. The fourth subfigure shows the trajectories
of concentration of mAb in mobile phase and last two subfigures depict
the adsorbed mAb. They shows that both binding sites of particles are
gradually approaching to their saturation limits.

In Figure `3.3 <#fig:down_state2>`__, the third subfigure represents
that the adsorbed mAb are washed out from the particles and transferred
into the mobile phase. Therefore, the trajectories of mAB in mobile
phase, the first subfigure, increase at the beginning. Then because the
capture column is connected to the VI loop, mAbs gradually transfer from
the capture column into the VI loop. This can be reflected in the second
subfigure. The trajectories of remaining unit operations can be found by
implementing the simulation.

.. container:: center

   .. figure:: imgs/down_x1_capture_level1_25_49.png
      :alt: State trajectories in capture column loading mode
      :name: fig:down_state1
      :width: 40.0%

      State trajectories in capture column loading mode

.. container:: center

   .. figure:: imgs/down_x2_elution_vi.png
      :alt: State trajectories in capture column elution mode
      :name: fig:down_state2
      :width: 40.0%

      State trajectories in capture column elution mode

Control problem formulation and controller design
=================================================

In this chapter, we present preliminary results of implementing advanced
process control (APC) techniques on the operation of the continuous mAb
production process. Specifically, two variants of APC algorithms namely
Model Predictive Control (MPC) and Economic Model Predictive Control
(EMPC) were designed and tested on the mAb production process. We begin
the chapter by presenting the control problem to be addressed.
Subsequently, we present the various controller designs. Finally, we
compare the results of MPC and EMPC.

Control problem formulation
---------------------------

Upstream process
~~~~~~~~~~~~~~~~

Before we begin this section, let us re-write the model of the upstream
mAb production process in the state space form

.. math::

   \label{eqn:state_space}
       \dot{x}(t) = f(x(t),u(t))

where :math:`\dot{x}(t) \in \mathbb R^{15}` is the velocity of the state
vector :math:`x \in \mathbb R^{15}` at time :math:`t` and
:math:`u(t)  \in \mathbb R^{7}` is the input vector. The variables in
the input vector will be defined later in this section. For practical
reasons, we assume that the state and input are constrained to be in the
spaces :math:`\mathbb X` and :math:`\mathbb U` respectively.

The primary control objective in this work is to ensure that safety and
environmental regulations are adhered to during the operation of the mAb
production process. From an economic point of view, it is essential to
maximize the production of mAb in the upstream process. Thus, two
secondary economic objectives are considered. The first is the
maximization of the mAb flow rate from the bioreactor while the second
is the maximization of the mAb flow rate in the separator
(microfiltration unit). These objectives are given as

.. math:: \ell_{\text{bioreactor}} = \text{mAb concentration in bioreactor} \times \text{flow out of the bioreactor}

.. math:: \ell_{\text{separator}} = \text{mAb concentration in separator} \times \text{flow out of the separator}

Combining the two economic objectives the following economic objective
is obtained.

.. math:: \ell_e(x,u) = \ell_{\text{bioreactor}} + \ell_{\text{separator}}

To achieve these objectives, the flow rates :math:`F_{in}`, :math:`F_r`,
:math:`F_1` and :math:`F_2`, the coolant temperature :math:`T_c`
together with the concentration of ammonia and glucose in the fresh
media stream are manipulated (input variables). Considering the
objectives, it is essential that advanced process control (APC)
algorithms which consider the complex system interaction while ensuring
constraint satisfaction are used.

Let us define the steady state economic optimization with respect to the
economic objective :math:`\ell_e` as

.. container:: subequations

   [eqn:ss_opt]

   .. math::

      \begin{aligned}
              (x_s,u_s) &= \arg \min ~ -\ell_e(x,u) \label{eqn:ss_opt_cost}\\
              s.t. ~~~ & 0 = f(x,u) \label{eqn:ss_opt_model}\\ 
              & x \in \mathbb{X} \label{eqn:ss_opt_state_con}\\
              & u \in \mathbb{U} \label{eqn:ss_opt_input_con}
          \end{aligned}

where Equation `[eqn:ss_opt_model] <#eqn:ss_opt_model>`__ is the system
model defined in Equation `[eqn:state_space] <#eqn:state_space>`__ with
zero state velocity, and
Equations `[eqn:ss_opt_state_con] <#eqn:ss_opt_state_con>`__ and 
`[eqn:ss_opt_input_con] <#eqn:ss_opt_input_con>`__ are the the
constraints on the state and the input respectively. The negative
economic cost function converts the maximization problem to a
minimization problem. The optimal value function in
`[eqn:ss_opt] <#eqn:ss_opt>`__ is used as the setpoint for MPC to track.

Controller design
-----------------

.. _sec:control_mpc:

Tracking Model Predictive Control (MPC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MPC is a multivariable advanced process control algorithm which has
gained significant attention in the process control community. This is
because of its ability to handle the complex system interactions and
constraints in the controller design. At each sampling time :math:`t_k`,
the following dynamic optimization problem is solved:

.. container:: subequations

   [eqn:mpc_opt]

   .. math::

      \begin{aligned}
              \min_{\bf{u}} & ~~~ \int_{t_k}^{t_k + N\Delta} (x(t)-x_s)^T Q (x(t)-x_s) + (u(t)-u_s)^T R (u(t)-u_s) dt \label{eqn:mpc_opt_a}\\
              s.t. & ~~~ \dot{x}(t) = f(x(t),v(t)) \label{eqn:mpc_opt_b} \\
              & ~~~ x(t_{k}) = x(t_{k}) \label{eqn:mpc_opt_c}\\
              & ~~~  x(t) \in \mathbb{X} \label{eqn:mpc_opt_d}\\
              & ~~~ u(t) \in \mathbb{U}  \label{eqn:mpc_opt_e}
          \end{aligned}

In the optimization problem `[eqn:mpc_opt] <#eqn:mpc_opt>`__ above,
Equation `[eqn:mpc_opt_b] <#eqn:mpc_opt_b>`__ is the model constraint
which is used to make predictions into the future,
Equation `[eqn:mpc_opt_c] <#eqn:mpc_opt_c>`__ is the initial state
constraint, :math:`\Delta` is the sampling time, :math:`N` is the
prediction and control horizons,
Equations `[eqn:mpc_opt_d] <#eqn:mpc_opt_d>`__
and `[eqn:mpc_opt_e] <#eqn:mpc_opt_e>`__ are the constraints on the
state and input respectively, and :math:`Q` and :math:`R` are matrices
of appropriate dimensions which represent the weights on the deviation
of states and the inputs from the setpoint. The setpoint is obtained by
solving the steady-state optimization problem
in `[eqn:empc_opt] <#eqn:empc_opt>`__. The decision variable **u**
in `[eqn:mpc_opt] <#eqn:mpc_opt>`__ is the optimal input sequence for
the process. The first input :math:`u(t_k)` is applied to the system and
the optimization problem is solved again after one sampling time.

Economic Model Predictive Control (EMPC)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MPC described in Section `4.2.1 <#sec:control_mpc>`__ uses a
quadratic cost in its formulation. However, in recent years MPC with a
general objective known as economic MPC (EMPC) has received significant
attention. The objective function in an EMPC generally reflects some
economic performance criterion such as profit maximization or heat
minimization. This is in contrast with the tracking MPC described
earlier where the objective is a positive definite quadratic function.
The integration of process economics directly in the control layer makes
EMPC of interest in many areas especially in the process industry. There
has been a significant number of applications of EMPC.

At each sampling time :math:`t_k`, the following optimization problem is
solved

.. container:: subequations

   [eqn:empc_opt]

   .. math::

      \begin{aligned}
              \min_{\bf{u}} & ~~~ \int_{t_k}^{t_k + N\Delta} -\ell_e(x(t),u(t)) dt \label{eqn:empc_opt_a}\\
              s.t. & ~~~ \dot{x}(t) = f(x(t),u(t)) \label{eqn:empc_opt_b} \\
              & ~~~ x(t_{k}) = x(t_{k}) \label{eqn:empc_opt_c}\\
              & ~~~  x(t) \in \mathbb{X} \label{eqn:empc_opt_d}\\
              & ~~~ u(t) \in \mathbb{U}  \label{eqn:empc_opt_e}
          \end{aligned}

In the optimization problem (`[eqn:empc_opt] <#eqn:empc_opt>`__) above,
the constraints are the same as the optimization problem in
(`[eqn:mpc_opt] <#eqn:mpc_opt>`__). However, a general cost function is
used in place of the quadratic cost function. The benefits of EMPC over
MPC will be demonstrated in the results section.

Simulation settings
-------------------

After conducting extensive open-loop tests, the control and prediction
horizons :math:`N` for both controllers was fixed at 100. This implies
that at a sampling time of 1 hour, the controllers plan 100 hours into
the future. The weights on the deviation of the states and input from
the setpoint were identify matrices. As mentioned earlier, the setpoint
for the tracking MPC was determined by solving the optimization problem
in `[eqn:ss_opt] <#eqn:ss_opt>`__.

.. _results-and-discussion-1:

Results and discussion
----------------------

The state and input trajectories of the system under the operation of
both MPC and EMPC is shown in Figures `4.1 <#fig:Figure_1>`__ and
`4.12 <#fig:Figure_12>`__. It can be seen that MPC and EMPC uses
different strategies to control the process. As an example, it can be
seen in Figure `4.9 <#fig:Figure_9>`__ that EMPC initially heats up the
system before gradually reducing it whereas MPC goes to the setpoint and
stays there. Again, EMPC tries to reduce the flow of the recycle stream
while MPC increases it as can be seen in Figure
`4.11 <#fig:Figure_11>`__. In both controllers though, the recycle
stream flow rate was kept low. Although the setpoint for the MPC was
determined under the same economic cost function used in the EMPC, it
can be seen that the EMPC does not go to that optimal steady state. This
could be due to the horizon being short for EMPC. Another possibility
could be due to numerical errors since the cost function was not scaled
in the EMPC. The cases where MPC was unable to go to the setpoint could
be due to numerical errors as a result of the large values of the states
and inputs. Further analysis may be required to confirm these
assertions.

.. container:: center

   .. figure:: imgs/Figure_1.png
      :alt: Trajectories of concentration of viable cells in the
      bioreactor and separator under the two control algorithms
      :name: fig:Figure_1
      :width: 90.0%

      Trajectories of concentration of viable cells in the bioreactor
      and separator under the two control algorithms

.. container:: center

   .. figure:: imgs/Figure_2.png
      :alt: Trajectories of total viable cells in the bioreactor and
      separator under the two control algorithms
      :name: fig:Figure_2
      :width: 90.0%

      Trajectories of total viable cells in the bioreactor and separator
      under the two control algorithms

.. container:: center

   .. figure:: imgs/Figure_3.png
      :alt: Trajectories of glucose concentration in the bioreactor and
      separator under the two control algorithms
      :name: fig:Figure_3
      :width: 90.0%

      Trajectories of glucose concentration in the bioreactor and
      separator under the two control algorithms

.. container:: center

   .. figure:: imgs/Figure_4.png
      :alt: Trajectories of glutamine concentration in the bioreactor
      and separator under the two control algorithms
      :name: fig:Figure_4
      :width: 90.0%

      Trajectories of glutamine concentration in the bioreactor and
      separator under the two control algorithms

.. container:: center

   .. figure:: imgs/Figure_5.png
      :alt: Trajectories of lactate concentration in the bioreactor and
      separator under the two control algorithms
      :name: fig:Figure_5
      :width: 90.0%

      Trajectories of lactate concentration in the bioreactor and
      separator under the two control algorithms

.. container:: center

   .. figure:: imgs/Figure_6.png
      :alt: Trajectories of ammonia concentration in the bioreactor and
      separator under the two control algorithms
      :name: fig:Figure_6
      :width: 90.0%

      Trajectories of ammonia concentration in the bioreactor and
      separator under the two control algorithms

.. container:: center

   .. figure:: imgs/Figure_7.png
      :alt: Trajectories of mAb concentration in the bioreactor and
      separator under the two control algorithms
      :name: fig:Figure_7
      :width: 90.0%

      Trajectories of mAb concentration in the bioreactor and separator
      under the two control algorithms

.. container:: center

   .. figure:: imgs/Figure_8.png
      :alt: Trajectories of reaction mixture volume in the bioreactor
      and separator under the two control algorithms
      :name: fig:Figure_8.png
      :width: 90.0%

      Trajectories of reaction mixture volume in the bioreactor and
      separator under the two control algorithms

.. container:: center

   .. figure:: imgs/Figure_9.png
      :alt: Trajectories of the bioreactor temperature and the coolant
      temperature under the two control algorithms
      :name: fig:Figure_9
      :width: 90.0%

      Trajectories of the bioreactor temperature and the coolant
      temperature under the two control algorithms

.. container:: center

   .. figure:: imgs/Figure_10.png
      :alt: Trajectories of flow in and out of the bioreactor under the
      two control algorithms
      :name: fig:Figure_10
      :width: 90.0%

      Trajectories of flow in and out of the bioreactor under the two
      control algorithms

.. container:: center

   .. figure:: imgs/Figure_11.png
      :alt: Trajectories of the recycle flow rate and the flow rate out
      of the upstream process under the two control algorithms
      :name: fig:Figure_11
      :width: 90.0%

      Trajectories of the recycle flow rate and the flow rate out of the
      upstream process under the two control algorithms

.. container:: center

   .. figure:: imgs/Figure_12.png
      :alt: Trajectories of glucose in fresh media under the two control
      algorithms
      :name: fig:Figure_12
      :width: 90.0%

      Trajectories of glucose in fresh media under the two control
      algorithms

Conclusion and future work
==========================

The integrated continuous manufacturing process of mAb brings
significant attentions since it is able to increase the productivity of
mAbs. From perspective of process control engineering, in order to study
the performance of APC on the process, the computational model of the
integrated continuous process is being neglected. In this work, the
first-principle models for all unit operations involved in the process
are studied and their computation models are constructed in Python. The
Python project serves as the digital twin of the integrated
manufacturing process. Due to the discontinuous nature of separation
columns involved in the downstream, a twin-column configuration at
capture step is implemented to achieve the continuous separation. It was
observed that EMPC and MPC used different strategies to control the
process with EMPC yielding a higher mAb concentration.

In the future, several APC topics are worth to be explored. Currently,
the integrated model is designed under the nominal case and assumes the
constant production from the upstream. However, this would not be the
case in reality. If less production is obtained from the upstream while
the time used to load the capture column is still the same, the capture
column will not be fully loaded. Hence with the appropriate measurement
or estimation of the production from the upstream, APC is able to
provide the appropriate time duration to load the capture column.

APC is also able to determine the best timing to switch between two
parallel columns. However, when the switching between columns is
involved in the APC, the nature of optimization problem becomes a
mixed-integer optimization problem that is more complicated and time
consuming to solve. With the help of the reinforcement learning (RL), a
hybrid APC algorithm that utilizes RL to tackle the issues caused by
integers, could be a potential solution.

Model parameters
================

.. _upstream-1:

Upstream
--------

.. container:: center

   .. container::
      :name: tb:upstream_parameters

      .. table:: Parameters for the upstream process model

         +----------------------+------------------------------------+-----------------------------+
         | Parameter            | Unit                               | Value                       |
         +======================+====================================+=============================+
         | :math:`K_{d,amm}`    | :math:`mM`                         | :math:`1.76`                |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`K_{d,gln}`    | :math:`hr^{-1}`                    | :math:`0.0096`              |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`K_{glc}`      | :math:`mM`                         | :math:`0.75`                |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`K_{gln}`      | :math:`mM`                         | :math:`0.038`               |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`KI_{amm}`     | :math:`mM`                         | :math:`28.48`               |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`KI_{lac}`     | :math:`mM`                         | :math:`171.76`              |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`m_{glc}`      | :math:`mmol/(cell \cdot hr)`       | :math:`4.9 \times 10^{-14}` |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`Q_{mAb}^{max}`| :math:`mg/(cell\cdot hr)`          | :math:`6.59 \times 10^{-10}`|
         +----------------------+------------------------------------+-----------------------------+
         | :math:`Y_{amm,gln}`  | :math:`mmol/mmol`                  | :math:`0.45`                |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`Y_{lac,glc}`  | :math:`mmol/mmol`                  | :math:`2.0`                 |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`Y_{X,glc}`    | :math:`cell/mmol`                  | :math:`2.6 \times 10^8`     |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`Y_{X,gln}`    | :math:`cell/mmol`                  | :math:`8.0 \times 10^8`     |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`\alpha_1`     | :math:`(mM \cdot L)/(cell \cdot h)`| :math:`3.4 \times 10^{-13}` |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`\alpha_2`     | :math:`mM`                         | 4.0                         |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`-\Delta H`    | :math:`J/mol`                      | :math:`5.0 \times 10^5`     |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`rho`          | :math:`g/L`                        | :math:`1560.0`              |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`c_p`          | :math:`J/(g ^\circ C)`             | :math:`1.244`               |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`U`            | :math:`J/(h ^\circ C)`             | :math:`4 \times 10^2`       |
         +----------------------+------------------------------------+-----------------------------+
         | :math:`T_{in}`       | :math:`^\circ C`                   | :math:`37.0`                |
         +----------------------+------------------------------------+-----------------------------+

.. _downstream-1:

Downstream
----------

The parameters of downstream model are obtained from the work of
Gomis-Fons et al :raw-latex:`\cite{gomis2020model}` and several
parameters are modified because the process is upscaled from lab scale
to industrial scale. They are summarized in
Table `6.2 <#tb:para_down>`__.

.. container:: center

   .. container::
      :name: tb:para_down

      .. table:: Parameters of digital twin of downstream

         +---------+-----------------------+---------------------+--------------------------------+
         | Step    | Parameter             | Unit                | Value                          |
         +=========+=======================+=====================+================================+
         | Capture | :math:`q_{max,1}`     | :math:`mg/mL`       | :math:`36.45`                  |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`k_{1}`         | :math:`mL/(mg~min)` | :math:`0.704`                  |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`q_{max,2}`     | :math:`mg/mL`       | :math:`77.85`                  |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`k_{2}`         | :math:`mL/(mg~min)` | :math:`2.1\cdot10^{-2}`        |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`K`             | :math:`mL/mg`       | :math:`15.3`                   |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`D_{eff}`       | :math:`cm^{2}/min`  | :math:`7.6\cdot10^{-5}`        |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`D_{ax}`        | :math:`cm^{2}/min`  | :math:`5.5\cdot10^{-1}v`       |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`k_{f}`         | :math:`cm/min`      | :math:`6.7\cdot10^{-2}v^{0.58}`|
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`r_{p}`         | :math:`cm`          | :math:`4.25\cdot10^{-3}`       |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`L`             | :math:`cm`          | :math:`20`                     |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`V`             | :math:`mL`          | :math:`10^5`                   |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`\epsilon_c`    | :math:`-`           | :math:`0.31`                   |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`\epsilon_p`    | :math:`-`           | :math:`0.94`                   |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`q_{max,elu}`   | :math:`mg/mL`       | :math:`114.3`                  |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`k_{elu}`       | :math:`min^{-1}`    | :math:`0.64`                   |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`H_{0,elu}`     | :math:`M^{\beta}`   | :math:`2.2\cdot10^{-2}`        |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`\beta_{elu}`   | :math:`-`           | :math:`0.2`                    |
         +---------+-----------------------+---------------------+--------------------------------+
         | Loop    | :math:`D_{ax}`        | :math:`cm^{2}/min`  | :math:`2.9\cdot10^{2}v`        |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`L`             | :math:`cm`          | :math:`600`                    |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`V`             | :math:`mL`          | :math:`5\cdot10^5`             |
         +---------+-----------------------+---------------------+--------------------------------+
         | CEX     | :math:`q_{max}`       | :math:`mg/mL`       | :math:`150.2`                  |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`k`             | :math:`min^{-1}`    | :math:`0.99`                   |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`H_{0}`         | :math:`M^{\beta}`   | :math:`6.9\cdot10^{-4}`        |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`\beta`         | :math:`-`           | :math:`8.5`                    |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`D_{app}`       | :math:`cm^{2}/min`  | :math:`1.1\cdot10^{-1}v`       |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`L`             | :math:`cm`          | :math:`10`                     |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`V`             | :math:`mL`          | :math:`5\cdot10^{4}`           |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`\epsilon_{c}`  | :math:`-`           | :math:`0.34`                   |
         +---------+-----------------------+---------------------+--------------------------------+
         | AEX     | :math:`D_{app}`       | :math:`cm^{2}/min`  | :math:`1.6\cdot10^{-1}v`       |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`k`             | :math:`min^{-1}`    | :math:`0`                      |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`L`             | :math:`cm`          | 10                             |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`V`             | :math:`mL`          | :math:`5\cdot10^{4}`           |
         +---------+-----------------------+---------------------+--------------------------------+
         |         | :math:`\epsilon_{c}`  | :math:`-`           | :math:`0.34`                   |
         +---------+-----------------------+---------------------+--------------------------------+
