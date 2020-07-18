# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import os

import numpy as np
import open3d as o3d
from pyntcloud import PyntCloud


# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    '''
    pca = PCA(n_components=1)
    pca.fit(data)
    #pca_score = pca.explained_variance_ratio_
    v = pca.components_
    l = pca.singular_values_


    :param data:
    :param correlation:
    :param sort:
    :return:
    '''

    x = data #N*3
    xMean = x.mean(0)
    #row = x.shape[0]
    '''
    xnormal =x.copy()
    for index in range(x.shape[0]):
        #print(index)
        xnormal[index] = x[index,:] - xMean
    '''
    xnormal=x-xMean
    H = np.dot(xnormal.transpose() , xnormal)
    eigenvectors, eigenvalues, vh = np.linalg.svd(H, full_matrices=True)
    #smat = np.diag(eigenvalues)
    #print(np.allclose(H, np.dot(eigenvectors, np.dot(smat, vh))))
    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    #save capture in this folder
    if not os.path.exists("../TestData/image/"):
        os.makedirs("../TestData/image/")

    # 指定点云路径
    cat_index = 39 # 物体编号，范围是0-39，即对应数据集中40个物体
    root_dir = '/home/ljn/SLAM/dateset/modelnet40_normal_resampled' # 数据集路径
    cat = os.listdir(root_dir)
    #for cat_index in range(len(cat)):# bug

    filename = os.path.join(root_dir, cat[cat_index], cat[cat_index]+'_0001.txt') # 默认使用第一个点云
    print(filename)
    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file(filename,sep=",",header=None,names=["x", "y", "z"],usecols=[0, 1, 2])
    #point_cloud_pynt = PyntCloud.from_file("/Users/renqian/Downloads/program/cloud_data/11.ply")#fun in pyntcloud
    #point_cloud_pd = pd.read_csv(filename, names=['x', 'y', 'z'], header=None,usecols=[0, 1, 2])#fun in pandas

    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    #o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    points = point_cloud_pynt.points
    print('total points number is:', points.shape[0])

    # 用PCA分析点云主方向
    w, v = PCA(points.values)
    point_cloud_vector0 = v[:, 0] #点云主方向对应的向量
    point_cloud_vector1 = v[:, 1]  # 点云次方向对应的向量
    print('the main orientation of this pointcloud is: ', point_cloud_vector0)
    # TODO: 此处只显示了点云，还没有显示PCA
    line_main_orient = o3d.geometry.LineSet()
    lines_points = np.array([-2*point_cloud_vector0, 2*point_cloud_vector0,\
                             -1*point_cloud_vector1,1*point_cloud_vector1], dtype=np.float32)
    lines = [[0, 1],[2, 3]]  # Right leg
    colors = [[1, 0, 0] ,[0, 0, 1]]  # main is red, second is blue
    line_main_orient.lines = o3d.utility.Vector2iVector(lines)
    line_main_orient.colors = o3d.utility.Vector3dVector(colors)
    line_main_orient.points = o3d.utility.Vector3dVector(lines_points)

    #o3d.visualization.draw_geometries([point_cloud_o3d])
    #o3d.visualization.draw_geometries([point_cloud_o3d,line_main_orient])
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D_PCA', width=860, height=540, left=50, top=50, visible=True)
    vis.add_geometry(point_cloud_o3d)
    vis.add_geometry(line_main_orient)
    filename = os.path.join('../TestData/image/', cat[cat_index] + '_pca_0001.png')
    vis.run()
    vis.capture_screen_image(filename, do_render=False)
    vis.destroy_window()

    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # 作业2
    # 屏蔽开始
    #point_cloud_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))#calculate normals using open3d
    # 由于最近邻搜索是第二章的内容，所以此处允许直接调用open3d中的函数
    for NormalIndex in range(points.shape[0]):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[NormalIndex], 50)
        #np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
        NeighborOfPoint = np.asarray(point_cloud_o3d.points)[idx[1:],:]
        Eigval, Eigvec = PCA(NeighborOfPoint)
        normal = Eigvec[:,2]
        normals.append(normal)

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    #o3d.visualization.draw_geometries([point_cloud_o3d],point_show_normal=True)

    vis = o3d.visualization.Visualizer()

    vis.create_window(window_name='Open3D_normals', width=860, height=540, left=50, top=50, visible=True)
    options = vis.get_render_option()
    options.point_show_normal = True
    vis.add_geometry(point_cloud_o3d)
    filename = os.path.join('../TestData/image/', cat[cat_index] + '_normals_0001.png')
    vis.run()
    vis.capture_screen_image(filename, do_render=False)
    vis.destroy_window()


if __name__ == '__main__':
    main()
