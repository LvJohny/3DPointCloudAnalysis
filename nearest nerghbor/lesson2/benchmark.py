# 对数据集中的点云，批量执行构建树和查找，包括kdtree和octree，并评测其运行时间

import random
import math
import numpy as np
import time
import os
import struct
from scipy import spatial

import octree as octree
import kdtree as kdtree
from result_set import KNNResultSet, RadiusNNResultSet

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

def main():
    # configuration
    leaf_size = 32
    min_extent = 0.0001
    k = 8
    radius = 1

    #root_dir = '/Users/renqian/cloud_lesson/kitti' # 数据集路径
    #cat = os.listdir(root_dir)
    #iteration_num = len(cat)
    # load date
    filename = '/home/ljn/SLAM/dateset/000000.bin'
    db_np = read_velodyne_bin(filename)
    iteration = 1



    print("octree -----------------------------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    result_set_knn = KNNResultSet(capacity=k)
    query = db_np[95, :]

    begin_t = time.time()
    root = octree.octree_construction(db_np, leaf_size, min_extent)
    construction_time_sum += time.time() - begin_t



    begin_t = time.time()

    octree.octree_knn_search(root, db_np, result_set_knn, query)
    knn_time_sum += time.time() - begin_t
    print('knn search result\n',result_set_knn)

    begin_t = time.time()
    result_set_rnn = RadiusNNResultSet(radius=radius)
    octree.octree_radius_search_fast(root, db_np, result_set_rnn, query)
    radius_time_sum += time.time() - begin_t
    print('rnn search result\n', result_set_rnn)

    begin_t = time.time()
    diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
    nn_idx = np.argsort(diff)
    nn_dist = diff[nn_idx]
    nn_dist_idx_pre = np.linspace(0,nn_dist.shape[0],nn_dist.shape[0])
    nn_dist_idx = nn_dist_idx_pre[nn_idx]
    brute_time_sum += time.time() - begin_t

    #brute knn search
    result_set_knn_brute = KNNResultSet(capacity=k)
    for index in range(k):
        result_set_knn_brute.add_point(nn_dist[index],nn_dist_idx[index])
    # brute radiusNN search
    result_set_rnn_brute = RadiusNNResultSet(radius=radius)
    for index in range(nn_dist.shape[0]):
        if nn_dist[index] < radius:
            result_set_rnn_brute.add_point(nn_dist[index], nn_dist_idx[index])
            continue
        else:
            break


    print("Octree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum*1000,
                                                                     knn_time_sum*1000,
                                                                     radius_time_sum*1000,
                                                                     brute_time_sum*1000))



    print("kdtree -----------------------------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0
    brute_time_sum = 0
    for i in range(iteration):
        query = db_np[95, :]

        begin_t = time.time()
        root = kdtree.kdtree_construction(db_np, leaf_size)
        construction_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set_knn = KNNResultSet(capacity=k)
        kdtree.kdtree_knn_search(root, db_np, result_set_knn, query)
        knn_time_sum += time.time() - begin_t

        begin_t = time.time()
        result_set_rnn = RadiusNNResultSet(radius=radius)
        kdtree.kdtree_radius_search(root, db_np, result_set_rnn, query)
        radius_time_sum += time.time() - begin_t

        begin_t = time.time()
        diff = np.linalg.norm(np.expand_dims(query, 0) - db_np, axis=1)
        nn_idx = np.argsort(diff)
        nn_dist = diff[nn_idx]
        brute_time_sum += time.time() - begin_t

        nn_dist_idx_pre = np.linspace(0,nn_dist.shape[0]-1,nn_dist.shape[0])
        nn_dist_idx = nn_dist_idx_pre[nn_idx]
        # brute knn search
        result_set_knn_brute = KNNResultSet(capacity=k)
        for index in range(k):
            result_set_knn_brute.add_point(nn_dist[index], nn_dist_idx[index])
        # brute radiusNN search
        result_set_rnn_brute = RadiusNNResultSet(radius=radius)
        for index in range(nn_dist.shape[0]):
            if nn_dist[index] < radius:
                result_set_rnn_brute.add_point(nn_dist[index], nn_dist_idx[index])
                continue
            else:
                break
    print("Kdtree: build %.3f, knn %.3f, radius %.3f, brute %.3f" % (construction_time_sum * 1000,
                                                                     knn_time_sum * 1000,
                                                                     radius_time_sum * 1000,
                                                                     brute_time_sum * 1000))

    print("scipy kdtree  -----------------------------------")
    construction_time_sum = 0
    knn_time_sum = 0
    radius_time_sum = 0

    query = db_np[95, :]

    begin_t = time.time()
    tree = spatial.KDTree(db_np, leaf_size)
    construction_time_sum += time.time() - begin_t

    #no knn
    begin_t = time.time()
    knn_time_sum += time.time() - begin_t


    begin_t = time.time()
    result_set_rnn = tree.query_ball_point(query, radius)
    radius_time_sum += time.time() - begin_t
    print('rnn search result\n', result_set_rnn)

    print("Octree: build %.3f, knn %.3f, radius %.3f" % (construction_time_sum * 1000,
                                                                     knn_time_sum * 1000,
                                                                     radius_time_sum * 1000))



if __name__ == '__main__':
    main()