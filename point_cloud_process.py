import numpy as np
import open3d
import sklearn
import time
from sklearn.mixture import GaussianMixture


if __name__ == '__main__':
    # pcd_load = open3d.io.read_point_cloud('./downpcd.pcd')
    pcd_load = open3d.io.read_point_cloud('./desity_cloud.pcd')
    pcd_array = np.asarray(pcd_load.points)
    pcd_z = pcd_array[:, 2]
    z_average = np.mean(pcd_z)
    pcd_HigherThanAver = pcd_array[np.where(pcd_z > z_average)]
    print(pcd_HigherThanAver.shape)
    pcd_1 = open3d.geometry.PointCloud()
    # pcd_2 = open3d.geometry.PointCloud()
    pcd_1.points = open3d.utility.Vector3dVector(pcd_HigherThanAver)
    # pcd_2.points = open3d.utility.Vector3dVector(pcd2)

    #
    # open3d.io.write_point_cloud("pcd_1.pcd", pcd_1)
    # open3d.io.write_point_cloud("pcd_2.pcd", pcd_2)
    # #
    # pcd_1 = open3d.io.read_point_cloud('./pcd_1.pcd')
    # pcd_2 = open3d.io.read_point_cloud('./pcd_2.pcd')
    open3d.visualization.draw_geometries([pcd_1], window_name="pcd")
    # print(pcd_z.shape)

    pass
