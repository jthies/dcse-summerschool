## Exercise 2

In this exercise we will consider the MaxCut problem and study its QUBO formulation and the equivalent Ising formulation

## Problem definition

The aim of the MaxCut problem is to find a partitioning of the vertices ($V$) of a graph $G=(V,E)$ into two disjoint sets $S$ and $T$, such that the number of edges ($E$) between $S$ and $T$ is as large as possible. Finding such a cut is known as the MaxCut problem.

### QUBO formulation of a small MaxCut problem

1. Open the [maximum_cut.py](maximum_cut.py) file and understand what it is doing
2. Run the QUBO formulation as follows:
   ```
   python maximum_cut.py
   ```
3. Switch to the D-Wave Problem Inspector and explore the solution
4. Modify the code to solve the MaxCut problem on another possibly larger graph.

### Ising model formulation of a small MaxCut problem

1. Repeat the above steps for the [maximum_cut_ising.py](maximum_cut_ising.py) file

### QUBO formulation of a larger MaxCut problem

1. Repeat the above steps for the [maximum_cut_large.py](maximum_cut_large.py) file

### Manual minor embedding

1. Open the [maximum_cut_minorembedding.py](maximum_cut_minorembedding.py) file and understand what it is doing
2. Run the QUBO formulation as follows:
   ```
   python maximum_cut_minorembedding).py
   ```
3. Override the minor embedding by uncommenting the line
   ```
   embedding = {1: [492], 2: [507], 3: [5237], 4: [522], 5: [5252]}
   ```
   and assigning physical qubits to the 5 binary variables `1`,...,`5`.
4. Change to another region and solver, e.g.
   ```
   sampler = DWaveSampler(region='eu-central-1', solver='Advantage_system5.3')
   ```
   If you have copied the provided [dwave.conf](../dwave.conf) you can easily switch between different profiles
   ```
   sampler = DWaveSampler(profile='eu')
   sampler = DWaveSampler(profile='na-1')
   [...]
   sampler = DWaveSampler(profile='na-4')
   ```
