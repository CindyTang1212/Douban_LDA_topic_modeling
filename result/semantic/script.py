# -*- coding: utf-8 -*-
# @Time    : 2021/11/23 6:43 PM
# @Author  : Cindy Tang
# @FileName: script.py
# @Software: PyCharm
# @Email    ：txdgz0624@gmail.com
from matplotlib import font_manager
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['ShiShangZhongHeiJianTi']
df = pd.read_csv('edgeTopic1_Final.csv')
G = nx.from_pandas_edgelist(df, edge_attr=True)
pos = nx.layout.spring_layout(G)
nx.draw(G, with_labels=True, edge_color='k', node_color='k', node_size=5, font_color='r', font_size=10, width=0.3, pos=pos)
plt.show()