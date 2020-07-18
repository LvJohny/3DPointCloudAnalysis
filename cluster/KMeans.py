# 文件功能： 实现 K-Means 算法

import numpy as np
from itertools import cycle, islice
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal

class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=3, tolerance=0.0001, max_iter=25):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        self.clusters = None
        self.r_nk = None


    def fit(self, data):
        # 作业1
        # 屏蔽开始
        min = data.min(axis=0)
        max = data.max(axis=0)
        rg = np.random.default_rng(1)
        cluster = rg.random((self.k_, 1)) * (max - min) + min

        # iteration
        for ite in range(self.max_iter_):
            # calculate r_nk
            r_nk = np.zeros((data.shape[0], self.k_), dtype=np.int16) # N*K
            for index in range(data.shape[0]):
                diff = np.linalg.norm(data[index, :] - cluster, axis=1)
                max = diff[0]
                max_idx = 0
                #找最小距离
                for i in range(diff.shape[0]-1):
                    if max > diff[i + 1]:
                        max = diff[i + 1]
                        max_idx = i + 1
                r_nk[index, max_idx] = 1
            sum_in_cluster = r_nk.sum(axis=0)
            if sum_in_cluster.all() == False:
                print('error: num_in_cluster,',sum_in_cluster)
                cluster = rg.random((self.k_, 1)) * (max - min) + min
                continue
            cluster = np.matmul(r_nk.T, data)
            cluster = (cluster.T * (1/sum_in_cluster)).T
        self.clusters = cluster
        self.r_nk = r_nk
        # 屏蔽结束

    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        for index in range(p_datas.shape[0]):
            for i in range(self.r_nk.shape[1]):
                if self.r_nk[index, i] == 1:
                    break
            result.append(i)
        # 屏蔽结束
        return result

def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
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
    X = np.array([[1, 2], [2, 2], [5, 8], [8, 8], [1, 0], [9, 11]])
    gmm = K_Means(n_clusters=2)
    gmm.fit(X)
    print(gmm)
    cat = gmm.predict(X)
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(cat) + 1))))
    plt.scatter(X[:, 0], X[:, 1], s=5,color = colors[cat])
    plt.show()
    print(cat)
    # 初始化