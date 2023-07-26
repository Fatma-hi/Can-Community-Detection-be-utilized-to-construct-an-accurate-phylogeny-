import networkx as nx
import sys
import csv
from networkx.drawing.nx_pydot import read_dot
from collections import defaultdict
from matplotlib import pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'figure.figsize': (15, 10)})
import random
import numpy as np
from numpy import random as nprand
import scipy.sparse as sparse
# from sknetwork.hierarchy import LouvainHierarchy
from scipy.cluster import hierarchy
import pandas as pd

T = pd.read_csv('7k.csv')
df = pd.DataFrame(T)
print(df.head())

print(df.shape)

x = df["seq"]
print(x.str.len().max())

for i in range(0,len(x)):
    x[i]=x[i].ljust(29892,'-')


def hammingDist(str1, str2):
    i = 0
    count = 0

    while (i < len(str1)):
        if (str1[i] != str2[i]):
            count += 1
        i += 1
    return count

EDGE = []
weights = []
matrix = np.zeros((len(x), len(x)))
for i in range(0, len(x)):
    for j in range(0, len(x)):
        matrix[i][j] = hammingDist(x[i], x[j])
        matrix[i][j] = 45 - (matrix[i][j])
        if matrix[i][j] > 35:
            if i != j:
                EDGE.append((i, j))
                weights.append(matrix[i][j])
                
with open('Edges.csv', 'w') as fp_out:
    fp_out.write('\n'.join(str(v) for v in EDGE))

with open('Weights.csv', 'w') as fp_out:
    fp_out.write('\n'.join(str(v) for v in weights))
