# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import numpy as np
from numpy.linalg import matrix_rank,det
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d

from cluster.Spectral_Clustering import  Spectral

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始

    # iteration
    ite = 120
    # point = [x, y, z, 1]
    data = np.column_stack((data, np.ones((data.shape[0], 1))))
    InliersData = data
    # select three points col plane
    Inlier_Point_is_few = True
    while ite > 0:
        if Inlier_Point_is_few == True:
            plane = InliersData[np.random.randint(InliersData.shape[0], size=3)]  # [0, b), plane 3*4
            if matrix_rank(plane) != 3:
                print('valid data: Point collinear')
                continue
            idx = np.array([[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]])
            PlaneRatios = np.array([det(plane[:, idx[0] - 1]), -det(plane[:, idx[1] - 1]),
                                    det(plane[:, idx[2] - 1]), -det(plane[:, idx[3] - 1])])

        pi = (PlaneRatios[0]**2+PlaneRatios[1]**2+PlaneRatios[2]**2)**0.5
        di = abs((PlaneRatios * data[:]).sum(axis=1)) / pi
        # sort inliers
        Inliers = (di < 0.1).tolist()
        InliersIdx = [i for i, x in enumerate(Inliers) if x]
        InliersData = data[InliersIdx]
        # refit plane 最小二乘解
        if InliersData.shape[0]/data.shape[0] > 0.3:
            if Inlier_Point_is_few:
                print('There are many inliers')
                Inlier_Point_is_few = False
            eigenvectors, eigenvalues, vh = np.linalg.svd(InliersData.T @ InliersData, full_matrices=True)
            sort = eigenvalues.argsort()
            eigenvalues = eigenvalues[sort]
            eigenvectors = eigenvectors[:, sort]
            PlaneRatios = eigenvectors[:,0]

        ite -= 1
    # end iteration

    inlier_points = InliersData[:, 0:3]
    Outliers = [i for i, x in enumerate(np.arange(data.shape[0])) if i not in InliersIdx]
    outlier_points = data[Outliers, 0:3]
    # 屏蔽结束

    print('ground point/total:', inlier_points.shape[0]/data.shape[0])
    return inlier_points,outlier_points

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始
    spectral = Spectral(n_clusters=3)
    spectral.fit(data)
    clusters_index = spectral.predict(data)

    # 屏蔽结束

    return clusters_index

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()

def main():
    root_dir = '/home/ljn/SLAM/dateset/KITTI 3D object detect' # 数据集路径
    cat = os.listdir(root_dir)
    iteration_num = len(cat)

    # vis = o3d.visualization.Visualizer()
    for i in range(iteration_num):
        filename = os.path.join(root_dir, cat[i])
        print('clustering pointcloud file:', filename)

        origin_points = read_velodyne_bin(filename)
        # origin visualization

        point_cloud_o3d = o3d.geometry.PointCloud()
        point_cloud_o3d.points = o3d.utility.Vector3dVector(origin_points)
        o3d.visualization.draw_geometries([point_cloud_o3d])

        ground_points, segmented_points = ground_segmentation(data=origin_points)
        ground_cloud, outlier_cloud = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        ground_cloud.points = o3d.utility.Vector3dVector(ground_points)
        outlier_cloud.points = o3d.utility.Vector3dVector(segmented_points)
        outlier_cloud.paint_uniform_color([1, 0, 0])
        ground_cloud.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([outlier_cloud,ground_cloud])

        # cluster
        cluster_index = clustering(segmented_points)
        plot_clusters(segmented_points, cluster_index)

if __name__ == '__main__':
    main()
