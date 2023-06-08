# ------ Import necessary packages ----
from collections import defaultdict

from neal import SimulatedAnnealingSampler
from dwave.system.samplers import DWaveSampler
from dwave.system.composites import EmbeddingComposite
import networkx as nx
import dwave.inspector

import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt

# ------- Set up our graph -------

# Create empty graph
G = nx.Graph()

# Add weighted edges to the graph (also adds nodes)
G.add_weighted_edges_from([(1,2,0.1),(1,3, 0.4),(1,4, 0.6),(2,3, 0.4),(2,4, 0.3),(3,4,0.3)])

# ------- Set up our QUBO dictionary -------

# Initialize our Q matrix
Q = defaultdict(int)

# you code goes here

# ------- Run our QUBO on the QPU -------
# Set up QPU parameters
chainstrength = 8
numruns = 10

# Run the QUBO on the solver from your config file
sampler = SimulatedAnnealingSampler
#sampler = EmbeddingComposite(DWaveSampler())
response = sampler.sample_qubo(Q,
                               chain_strength=chainstrength,
                               num_reads=numruns,
                               label='Example - TSP')

dwave.inspector.show(response)

# ------- Print results to user -------
# your code goes here
