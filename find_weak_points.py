import pandas as pd
import glob
import os
from pgmpy.models import BayesianNetwork

# Load edge data
edges_df = pd.read_excel('output/edges.xlsx')

# Create a BayesianModel object
model = BayesianNetwork(list(map(tuple, edges_df.values)))

# Convert to a networkx graph
nx_graph = model.to_directed()

# Find nodes that have multiple parent nodes
nodes_with_multiple_parents = [node for node in nx_graph.nodes() if len(list(nx_graph.predecessors(node))) > 1]

# # Print out these nodes
# print("Nodes with multiple parents:", nodes_with_multiple_parents)

# Save results to CSV file
with open('output/nodes_with_multiple_parents.csv', 'w') as f:
    for node in nodes_with_multiple_parents:
        f.write(node + '\n')

# Get all CPD filenames
cpd_files = glob.glob('output/cpds/*.xlsx')

# Find nodes that have high probability variance when given parent node states
nodes_with_high_prob_variance = []
for cpd_file in cpd_files:
    cpd_df = pd.read_excel(cpd_file)
    prob_column = cpd_df.columns[0]
    if cpd_df[prob_column].std() > 0.1:  # Set a threshold, adjust according to the situation
        node_name = os.path.basename(cpd_file).split('.')[0]
        nodes_with_high_prob_variance.append(node_name)

# # Print out these nodes
# print("Nodes with high probability variance:", nodes_with_high_prob_variance)

# Save results to CSV file
with open('output/nodes_with_high_prob_variance.csv', 'w') as f:
    for node in nodes_with_high_prob_variance:
        f.write(node + '\n')