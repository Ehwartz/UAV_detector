import open3d as o3d
import numpy as np
import copy
import time

# pcd =o3d.io.read_point_cloud('cloud7707e77a895eb400.pcd')
# print(pcd)
# voxel_size = 0.5
# downpcd = pcd.voxel_down_sample(voxel_size)
# o3d.io.write_point_cloud("downpcd.pcd", downpcd, True)
# print(downpcd)
downpcd = o3d.io.read_point_cloud('downpcd.pcd')
print(downpcd)
downpcd.paint_uniform_color([1, 0.706, 0])
downpcdcolor = np.asarray(downpcd.colors)


# o3d.visualization.draw_geometries([downpcd], window_name="可视化参数设置")


# def project_xy(pcd): #pcd is np
#     pcd[:,2]=0
#     return pcd
# points =np.asarray(downpcd.points)
# points_xy = project_xy(points)
# print(points_xy)

def density_cloud(pcd, step, th):  # points中是np的数组,网格密度统计
    points = np.asarray(pcd.points)
    x_min, y_min, z_min = np.amin(points, axis=0)
    x_max, y_max, z_max = np.amax(points, axis=0)
    width = np.ceil((x_max - x_min) / step)
    height = np.ceil((y_max - y_min) / step)
    M = np.ones((int(width), int(height)))
    for i in range(len(points)):
        row = np.floor((points[i][0] - x_min) / step)
        col = np.floor((points[i][1] - y_min) / step)
        M[int(row), int(col)] += 1
    ind = list()
    for i in range(len(points)):
        row = np.floor((points[i][0] - x_min) / step)
        col = np.floor((points[i][1] - y_min) / step)
        if M[int(row), int(col)] > th:
            ind.append(i)
    dcloud = pcd.select_by_index(ind)
    return dcloud


def density_cl(pcd, step, th):
    points = np.asarray(pcd.points)
    p_min = np.min(points, axis=0)
    rcs = np.floor((points[:, 0:2] - p_min[0:2]) / step).astype(np.int32)
    uniq, indices, counts = np.unique(rcs, axis=0, return_inverse=True, return_counts=True)
    mask = (counts>th)
    return points[mask[indices]]


if __name__ == '__main__':
    t0 = time.time()
    density_points = density_cloud(downpcd, 0.5, 30)
    t1 = time.time()
    print('--------------')


    # o3d.io.write_point_cloud("desity_cloud.pcd", density_points)
    o3d.visualization.draw_geometries([density_points], window_name="density_cloud")

    t2 = time.time()
    dp_arr = density_cl(downpcd, 0.5, 30)
    pcd = o3d.geometry.PointCloud()
    t3 = time.time()
    pcd.points = o3d.utility.Vector3dVector(dp_arr)

    # o3d.io.write_point_cloud("desity_cloud.pcd", density_points)
    o3d.visualization.draw_geometries([pcd], window_name="density_cloud2")

    print('t1: ', t1-t0)
    print('t2: ', t3-t2)
