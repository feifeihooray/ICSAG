import pandas as pd
from pgmpy.models import BayesianNetwork
import pygraphviz as pgv
from IPython.display import Image


# 读取边的数据
edges_df = pd.read_excel('output/edges.xlsx')

# 创建BayesianModel对象
model = BayesianNetwork(list(map(tuple, edges_df.values)))


# 转换为networkx图
nx_graph = model.to_directed()


# 创建一个新的有向图
G = pgv.AGraph(directed=True)

# 添加节点和边
for edge in model.edges():
    G.add_edge(edge[0], edge[1])

# 保存图像
G.draw('output/graph.png', prog='dot', format='png')

# 显示图像
Image(filename='output/graph.png')

