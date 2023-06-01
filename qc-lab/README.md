# QC Lab: First steps on a Quantum Annealer

In this lab we will learn the first steps to solve combinatorial optimization problems on a quantum annealer. In particular, we will see how certain combinatorial optimization problems can be formulated as a Quadratic Unconstrained Binary Optimizaiton (QUBO) problem of the form
$$\min_{x\in\\{0,1\\}^n} x^\top Q x = \sum_{i=1}^n\sum_{j=i}^nQ_{ij}x_ix_j$$
with real-valued upper triangular matrix $Q\in\mathbb{R}^{n\times n}$

## What you will practice here
- setting up D-Wave's Ocean SDK and running a first QUBO for the MaxCut problem
- using D-Wave's problem inspector to 'debug' the computation
- developing a QUBO formulation for the traveling salesperson from scratch

## Your tasks

1. __Create a free D-Wave Leap account__

   Create a free account at https://cloud.dwavesys.com/leap/

2. __Install and setup D-Wave Ocean SDK__

   Follow the instructions given at https://docs.ocean.dwavesys.com/en/stable/overview/install.html
   
   In esence you execute the following steps:
   
   - Create a Python virtual environment
     ```
     python -m venv ocean
     . ocean/bin/activate
     ```
     
   - Install the Ocean SDK
     ```
     pip install dwave-ocean-sdk
     ```
     
   - Setup your configuration
     ```
     $ dwave setup

     Optionally install non-open-source packages and configure your environment.

     Do you want to select non-open-source packages to install (y/n)? [y]: ↵

     D-Wave Drivers
     These drivers enable some automated performance-tuning features.
     This package is available under the 'D-Wave EULA' license.
     The terms of the license are available online: https://docs.ocean.dwavesys.com/eula
     Install (y/n)? [y]: ↵
     Installing: D-Wave Drivers
     Successfully installed D-Wave Drivers.

     D-Wave Problem Inspector
     This tool visualizes problems submitted to the quantum computer and the results returned.
     This package is available under the 'D-Wave EULA' license.
     The terms of the license are available online: https://docs.ocean.dwavesys.com/eula
     Install (y/n)? [y]: ↵
     Installing: D-Wave Problem Inspector
     Successfully installed D-Wave Problem Inspector.

     Creating the D-Wave configuration file.
     Using the simplified configuration flow.
     Try 'dwave config create --full' for more options.

     Creating new configuration file: /home/jane/.config/dwave/dwave.conf
     Profile [defaults]: ↵
     Updating existing profile: defaults
     Authentication token [skip]: ABC-1234567890abcdef1234567890abcdef ↵
     Configuration saved.
     ```
     
   - Overwrite `dwave.conf` file

     We suggest that you override the just created `dwave.conf` file by the one from this repository filling in the authentication token.
