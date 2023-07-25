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

# 打印节点的条件概率分布并保存到txt文件
with open('output/cpds.txt', 'w') as f:
    for cpd in model.get_cpds():
        f.write("CPD of {variable}:\n".format(variable=cpd.variable))
        f.write(str(cpd))
        f.write('\n\n')


# 保存我们学习到的模型的边到Excel文件
edges_df = pd.DataFrame(best_model.edges(), columns=['Parent_Node', 'Child_Node'])
edges_df.to_excel('output/edges.xlsx', index=False)

# 保存节点的条件概率分布到Excel文件
for cpd in model.get_cpds():
    # 创建一个空的DataFrame来存储这个CPD的数据
    cpd_df = pd.DataFrame(columns=[cpd.variable] + [f'Parent_{i+1}' for i in range(len(cpd.variables)-1)] + ['State', 'Probability'])
    # 遍历这个CPD的每个元素
    for indices in np.ndindex(*cpd.values.shape):
        # 获取对应的节点状态和概率值
        states = [cpd.state_names[var][index] for var, index in zip(cpd.variables, indices)]
        prob = cpd.values[indices]
        # 将这些数据添加到我们的DataFrame中
        cpd_df.loc[len(cpd_df)] = states + [prob]
    # 为每个节点保存一个单独的文件
    cpd_df.to_excel(f'output/cpds/{cpd.variable}.xlsx', index=False)
