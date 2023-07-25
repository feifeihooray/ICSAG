import pandas as pd
from pgmpy.models import BayesianNetwork

# 读取边的数据
edges_df = pd.read_excel('output/edges.xlsx')

# 创建BayesianModel对象
model = BayesianNetwork(list(map(tuple, edges_df.values)))

# 转换为networkx图
nx_graph = model.to_directed()

# 寻找具有多个父节点的节点
nodes_with_multiple_parents = [node for node in nx_graph.nodes() if len(list(nx_graph.predecessors(node))) > 1]

# 输出这些节点
print("Nodes with multiple parents:", nodes_with_multiple_parents)

# 读取节点的条件概率分布数据
cpds_df = pd.read_excel('output/cpds.xlsx')

# 寻找在给定父节点状态时有大概率变化的节点
nodes_with_high_prob_variance = []
for node in cpds_df.iloc[:, 0].unique():  # 我假设在你的'cpds.xlsx'中，每个节点名都是存储在第一列的
    node_cpds = cpds_df[cpds_df.iloc[:, 0] == node]
    if node_cpds['probability'].std() > 0.1:  # 设定阈值，根据情况调整
        nodes_with_high_prob_variance.append(node)

# 输出这些节点
print("Nodes with high probability variance:", nodes_with_high_prob_variance)
