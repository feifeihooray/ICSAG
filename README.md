# causal_graphs
1) Check how you can learn causal graphs from industrial control systems data.
   
   a. Data Importation: The data from the SWaT was read using the pandas library's read_excel() function.

   b. Causal Structure Learning: Use the HillClimbSearch and BicScore classes from the pgmpy library. HillClimbSearch is a heuristic search algorithm for determining the optimal structure of a Bayesian network, while BicScore is a scoring function based on the Bayesian Information Criterion (BIC) used to evaluate the fitness of the model. The resulting model was a directed acyclic graph (DAG), where each node represents a variable from the system, and each directed edge represents a causal relationship.
   
   c. Parameter Learning: Learn the parameters using the MaximumLikelihoodEstimator from pgmpy. This filled in the conditional probability tables (CPTs) for each node in the graph, specifying the probability of each state given the states of its parent nodes.
   
   d. Data Exportation: The learned causal graph (both the structure and the parameters) was then exported to Excel files for further examination. This included an Excel file listing the edges of the graph, and another file detailing the conditional probability distributions.
   
   e. Graph Visualization: Visualized the causal graph using pygraphviz and IPython's Image. This involved creating a new directed graph, adding nodes and edges according to our model, drawing and saving the graph as a PNG file, and finally displaying it.


3) how can you use these graphs for risk assessment in industrial control systems? 

   a. Identify Key Indicators: First, you would need to identify key indicators that could impact system performance. These could include the physical state of equipment (such as temperature, pressure, etc.), operational conditions of the system, and even external factors like environmental conditions.
   
   b. Build the Causal Graph: Next, you would build a causal graph that represents the relationships between these indicators. You could either use the causal graph you've already learned from data, or build one based on your understanding of the system.
   
   c. Quantify the Causal Graph: Then, you would assign a conditional probability to each causal relationship in the graph. This could be based on historical data or expert knowledge.
   
   d. Perform Inference: Within the causal graph, given some observed conditions, you can infer the state of other unobserved variables. For example, if you observe that a piece of equipment is overheating, you can infer the likelihood that this might lead to a system disruption.
   
   e. Assess Risk: Finally, based on the results of your inference, you can assess the risk of various issues. For example, you could assess the risk of a system disruption or the risk of equipment failure.
   
