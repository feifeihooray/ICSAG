import pandas as pd
from pgmpy.models import BayesianNetwork
import pygraphviz as pgv
from IPython.display import Image

# Load the edge data
edges_df = pd.read_excel('output/edges.xlsx')

# Create a BayesianModel object
model = BayesianNetwork(list(map(tuple, edges_df.values)))

# Convert to networkx graph
nx_graph = model.to_directed()

# Create a new directed graph
G = pgv.AGraph(directed=True)

# Add nodes and edges
for edge in model.edges():
    G.add_edge(edge[0], edge[1])

# Save the image
G.draw('output/graph.png', prog='dot', format='png')

# Display
Image(filename='output/graph.png')
