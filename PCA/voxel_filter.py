# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
import random

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
approximate = 'centroid'
#approximate = 'random'
def voxel_filter(point_cloud, leaf_size):

    filtered_points = []
    # 作业3
    # 屏蔽开始


    point_original = point_cloud.values

    min = point_original.min(axis=0)  # xmin = min[0]; ymin = min[1]; zmin = min[2]
    max = point_original.max(axis=0)

    #calculate leaf_radius
    lenXYZ = max -min
    leaf_radius = ((lenXYZ[0]*lenXYZ[1] + lenXYZ[1]*lenXYZ[2] +lenXYZ[0]*lenXYZ[2])/leaf_size)**0.5
    print('leaf_radius:',leaf_radius)
    D = lenXYZ / leaf_radius


    Ha = []
    for index in range(point_original.shape[0]):
        h = (point_original[index,:] - min) // leaf_radius  + 1
        H = h[0] + h[1]*D[0] + h[2]*D[0]*D[1]
        Ha.append(H)

    H_sort = np.asarray(Ha,dtype=np.float64)
    sort = H_sort.argsort()[::-1]
    H_sorted = H_sort[sort]
    point_original = point_original[sort, :]
    filter_set_sum = point_original[0,:]
    for index in range(point_original.shape[0]-1):
        if(H_sorted[index] == H_sorted[index+1]):
            filter_set_sum =np.vstack((filter_set_sum , point_original[index+1, :]))
        else:
            if (filter_set_sum.shape == (3,)):
                filtered_points.append(filter_set_sum)
            else:
                if (approximate == 'centroid'):
                    filtered_points.append(filter_set_sum.mean(0))
                elif(approximate == 'random'):
                    filtered_points.append(filter_set_sum[random.randint(0,filter_set_sum.shape[0]-1)])
                else:
                    print("error")
            filter_set_sum = point_original[index+1, :]
    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    print('point num:',filtered_points.shape)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    cat_index = 0 # 物体编号，范围是0-39，即对应数据集中40个物体 #
    root_dir = '/home/ljn/SLAM/dateset/modelnet40_normal_resampled'  # 数据集路径
    cat = os.listdir(root_dir)
    #['cone', 'bench', 'vase', 'toilet', 'bottle', 'sofa',
    # 'stairs', 'flower_pot', 'bathtub', 'piano', 'airplane',
    # 'bed', 'curtain', 'chair', 'mantel', 'keyboard', 'night_stand',
    # 'wardrobe', 'desk', 'table', 'door', 'dresser', 'car', 'laptop',
    # 'guitar', 'cup', 'glass_box', 'xbox', 'monitor', 'plant', 'tent',
    # 'bowl', 'radio', 'stool', 'person', 'tv_stand', 'range_hood', 'sink',
    # 'bookshelf', 'lamp']
    filename = os.path.join(root_dir, cat[cat_index], cat[cat_index] + '_0001.txt')  # 默认使用第一个点云
    # 加载原始点云
    point_cloud_pynt = PyntCloud.from_file(filename, sep=",", header=None, names=["x", "y", "z"], usecols=[0, 1, 2])

    # 转成open3d能识别的格式
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(point_cloud_pynt.points, 1000)# 采样后数量约为1000
    point_cloud_o3d  = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云 (更改)
    #o3d.visualization.draw_geometries([point_cloud_o3d])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Open3D_vexel_filter', width=860, height=540, left=50, top=50, visible=True)
    vis.add_geometry(point_cloud_o3d)
    filename = os.path.join('../TestData/image/', cat[cat_index] + '_vexel_filter'+approximate+'_0001.png')
    vis.run()
    vis.capture_screen_image(filename, do_render=False)
    vis.destroy_window()



if __name__ == '__main__':
    main()
