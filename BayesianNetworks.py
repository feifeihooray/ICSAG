import numpy as np
import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator

# 读取Excel文件数据
data = pd.read_excel('data/22June2020(1).xlsx', engine='openpyxl')

# 使用HillClimbSearch和BicScore来学习贝叶斯网络的结构
bic = BicScore(data)
hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=bic)


# 打印出我们学习到的模型的边
print(best_model.edges())

# 你可以使用最大似然估计法来估计条件概率分布
model = BayesianModel(best_model.edges())
model.fit(data, estimator=MaximumLikelihoodEstimator)

# 打印节点的条件概率分布
for cpd in model.get_cpds():
    print("CPD of {variable}:".format(variable=cpd.variable))
    print(cpd)

# 保存我们学习到的模型的边到Excel文件
edges_df = pd.DataFrame(best_model.edges(), columns=['Node1', 'Node2'])
edges_df.to_excel('output/edges.xlsx', index=False)

# 保存节点的条件概率分布到Excel文件
cpds = []
for cpd in model.get_cpds():
    # 创建一个空的DataFrame来存储这个CPD的数据
    cpd_df = pd.DataFrame(columns=[cpd.variable] + cpd.variables[1:] + ['probability'])
    # 遍历这个CPD的每个元素
    for indices in np.ndindex(*cpd.values.shape):
        # 获取对应的父节点状态和概率值
        parent_states = [cpd.state_names[var][index] for var, index in zip(cpd.variables, indices)]
        prob = cpd.values[indices]
        # 将这些数据添加到我们的DataFrame中
        cpd_df.loc[len(cpd_df)] = parent_states + [prob]
    cpds.append(cpd_df)

cpds_df = pd.concat(cpds)
cpds_df.to_excel('output/cpds.xlsx')
