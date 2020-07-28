# 文件功能：实现 Spectral Cluster 算法

import numpy as np
from numpy import *
import pylab
import random, math
from itertools import cycle, islice
import nearset_nerghbor.kdtree as kdtree
import nearset_nerghbor.octree as octree
from nearset_nerghbor.result_set import KNNResultSet, RadiusNNResultSet
import matplotlib.pyplot as plt
from itertools import chain,islice
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
from cluster.KMeans import K_Means
plt.style.use('seaborn')



class DBSACN(object):
    def __init__(self):
        self.n_clusters = None
        self.result = None

    def fit(self, data):
        # 作业3
        # 屏蔽开始
        # initiation
        # 参数
        min_samples = 3
        root = kdtree.kdtree_construction(data, leaf_size=10)
        leaf_size = ((data.max(axis=0) - data.min(axis=0)) / ((data.shape[0]) ** (0.52))).max()
        result = []
        noise = np.zeros((data.shape[0])).tolist()
        RemainderPoint = np.arange(data.shape[0]).tolist()
        index = 0
        while (index < data.shape[0]):
            # RNN
            result_set_i = RadiusNNResultSet(radius=leaf_size)
            kdtree.kdtree_radius_search(root, data, result_set_i, data[index])
            if(result_set_i.count < min_samples):
                noise[index] = 1
                RemainderPoint.remove(index) # visited
            result.append(result_set_i)
            index += 1
        # random select one point
        cluster_idx = 2
        while(noise.count(0) != 0):
            point_idx = RemainderPoint[np.random.randint(len(RemainderPoint))]
            RemainderPoint.remove(point_idx) # 从剩余点中剔除
            noise[point_idx] = cluster_idx # 标记为cluster

            cluster_idx_set =[point_idx]
            # 寻找该点的RNN,放如类别的cluster_idx_set里
            index_RNN= np.arange(result[point_idx].count).tolist()
            cluster_idx_set += [result[point_idx].dist_index_list[x].index for x in index_RNN
                                if ((noise[result[point_idx].dist_index_list[x].index] != 1) and
                                (result[point_idx].dist_index_list[x].index not in cluster_idx_set))]
            set_idx = 1
            while(True):
                if set_idx >= len(cluster_idx_set):
                    break
                point_idx = cluster_idx_set[set_idx]
                RemainderPoint.remove(point_idx)  # 从剩余点中剔除
                noise[point_idx] = cluster_idx  # 标记为cluster
                index_RNN = np.arange(result[point_idx].count).tolist()
                cluster_idx_set += [result[point_idx].dist_index_list[x].index for x in index_RNN
                                    if (noise[result[point_idx].dist_index_list[x].index] != 1) and
                                    (result[point_idx].dist_index_list[x].index not in cluster_idx_set)]
                set_idx += 1

            cluster_idx_set.clear()
            cluster_idx += 1


        self.result = noise

        # 屏蔽结束

    def predict(self, data):
        # 屏蔽开始
        result = self.result
        return result
        # 屏蔽结束


# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 200, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 100, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 300, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    # plt.figure(figsize=(10, 8))
    # plt.axis([-10, 15, -5, 15])
    # plt.scatter(X1[:, 0], X1[:, 1], s=5)
    # plt.scatter(X2[:, 0], X2[:, 1], s=5)
    # plt.scatter(X3[:, 0], X3[:, 1], s=5)
    # plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)
    #X =  np.array([[1, 2], [2, 2], [5, 8], [8, 8], [1, 0], [9, 11]])

    dbsacn = DBSACN()
    dbsacn.fit(X)
    cat = dbsacn.predict(X)
    print(cat)

    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(cat) + 1))))
    plt.scatter(X[:, 0], X[:, 1], s=5, color=colors[cat])
    plt.show()
    print(cat)
    # 初始化



