import numpy as np
import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

# Load the Excel file data
data = pd.read_excel('data/22June2020(2).xlsx', engine='openpyxl')

# Use HillClimbSearch and BicScore to learn the structure of the Bayesian network
bic = BicScore(data)
hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=bic)

# Use the maximum likelihood estimator to estimate the conditional probability distribution
model = BayesianModel(best_model.edges())
model.fit(data, estimator=MaximumLikelihoodEstimator)

# #Print the conditional probability distributions of nodes and save to a txt file
# with open('output/cpds.txt', 'w') as f:
#     for cpd in model.get_cpds():
#         f.write("CPD of {variable}:\n".format(variable=cpd.variable))
#         f.write(str(cpd))
#         f.write('\n\n')


# Save the edges of the learned model to an Excel file
edges_df = pd.DataFrame(best_model.edges(), columns=['Parent_Node', 'Child_Node'])
edges_df.to_excel('output/edges.xlsx', index=False)

for cpd in model.get_cpds():
    # print("CPD variables: ", cpd.variables)
    # print("CPD state names: ", cpd.state_names)
    # print("CPD values shape: ", cpd.values.shape)
    parent_name = cpd.variables[1] if len(cpd.variables) > 1 else None
    if parent_name:
        cpd_df = pd.DataFrame(columns=[cpd.variable, f'{parent_name}_State_1', f'{parent_name}_State_2'])
    else:
        cpd_df = pd.DataFrame(columns=[cpd.variable, 'Probability'])

    for indices in np.ndindex(*cpd.values.shape):
        states = [cpd.state_names[cpd.variable][indices[0]]] + [cpd.state_names[var][index] for var, index in zip(cpd.variables[1:], indices[1:])]
        prob = cpd.values[indices]

        if parent_name:
            new_row = {cpd_df.columns[0]: states[0], cpd_df.columns[1]: prob if indices[1] == 0 else 0, cpd_df.columns[2]: prob if indices[1] == 1 else 0}
        else:
            new_row = {cpd_df.columns[0]: states[0], cpd_df.columns[1]: prob}
        cpd_df = cpd_df.append(new_row, ignore_index=True)

    cpd_df.to_excel(f'output/cpds/{cpd.variable}.xlsx', index=False)
