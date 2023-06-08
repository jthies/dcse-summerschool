## Exercise 3

In this exercise we will develop a QUBO formulation for the traveling salesperson problem from scratch

## Problem definition

Given a list of cities, i.e. the vertices $V$ of the graph $G=(V,E)$, and the distances between each pair of cities, i.e. the edges $E$ of the graph $G$, compute the shortest possible route that visits each city exactly once and returns to the origin city.

### Derivation of the QUBO formulation 

1. What is the objective function, i.e. which $f(x_1, ...,x_n)$ should be minimized?

2. Think about ways to implement the following constraints in terms of the binary variables $x_1, ...,x_n$:
   - the route must pass though all cities
   - each city must be visited exactly once
   - start and endpoint of the route must be the same (you can specify the start/endpoint explicitly for simplicity)

3. Formulate a first QUBO with parametrized penalty/reward terms for the different constraints. Think about corner cases.
   - Does an intentional violation of constraints lead to a state that is an energetically favorable over the best admissible state? If so, how do you need to change the penalty/reward parameters to make *all* non-admissible states energetically less favorable? Start with a small 3-city example and explore all possibilities manually. Do the parameters scale for larger problem sizes? 

### Implementing and testing of the QUBO formulation

1. Start from the file [tsp.py](tsp.py) to implement the derived QUBO formulation.
2. Test your implementation first with the simulated annealing sampler (`sampler = SimulatedAnnealingSampler`) and then with D-Wave's hardware sampler (`sampler = EmbeddingComposite(DWaveSampler())`).
3. Compare your implementation with D-Wave's [TSP solver](https://docs.ocean.dwavesys.com/projects/dwave-networkx/en/latest/reference/algorithms/generated/dwave_networkx.algorithms.tsp.traveling_salesperson_qubo.html#dwave_networkx.algorithms.tsp.traveling_salesperson_qubo).
