# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

class GMM(object):
    def __init__(self, n_clusters, max_iter=25):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        self.gama_nk = None
        self.pi = None
        self.mu = None
        self.sigma = None

    def fit(self, data):
        # 作业3
        # 屏蔽开始
        #initiation
        min = data.min(axis=0)
        max = data.max(axis=0)
        rg = np.random.default_rng(1)
        mu = rg.random((self.n_clusters, 1)) * (max - min) + min
        sigma = np.ones((self.n_clusters,data.shape[1],data.shape[1]))
        sigma[:] *= np.eye(data.shape[1])
        pi = np.empty( (self.n_clusters) )
        pi[:] = 1/self.n_clusters
        #iteration
        for index in range(self.max_iter):
            #gama_nk K*N
            gama_nk = np.zeros((self.n_clusters , data.shape[0] ), dtype=np.float64)
            for ind_clu in range(self.n_clusters):
                for idx_point in range(data.shape[0]):
                    temp_up =pi[ind_clu] * multivariate_normal.pdf(data[idx_point], mean=mu[ind_clu], cov=sigma[ind_clu])
                    temp_down = 0
                    for temp_idx in range(self.n_clusters):
                        temp_down += pi[temp_idx] * multivariate_normal.pdf(data[idx_point], mean=mu[temp_idx], cov=sigma[temp_idx])
                    gama_nk[ind_clu, idx_point] = temp_up / temp_down
            N_k = gama_nk.sum(axis=1)
            # mu new
            mu = ((gama_nk @ data).T / N_k).T
            #sigma new
            for ind_clu in range(self.n_clusters):
                temp = 0
                for idx_point in range(data.shape[0]):
                    temp +=  gama_nk[ind_clu, idx_point] * (data[idx_point] - mu[ind_clu]).reshape(data.shape[1],1) @ (data[idx_point] - mu[ind_clu]).reshape(1,data.shape[1])
                sigma[ind_clu] = temp / N_k[ind_clu]
        #pi new
        pi = N_k / data.shape[0]
        # 更新gama_nk
        self.gama_nk = gama_nk
        # 更新pi
        self.pi = pi
        # 更新Mu
        self.mu = mu
        # 更新Var
        self.sigma = sigma

        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        result = []
        for index in range(self.gama_nk.shape[1]):
            max = self.gama_nk[0,index]
            max_idx = 0
            for i in range(self.gama_nk.shape[0] - 1):
                if max < self.gama_nk[i+1,index]:
                    max = self.gama_nk[i+1, index]
                    max_idx = i+1
            result.append(max_idx)
        return result
        # 屏蔽结束


# 生成仿真数据
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
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化

    

