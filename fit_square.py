import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
import sklearn
from sklearn.cluster import KMeans


def pcd2array_xy(pcd):
    arr = np.asarray(pcd.points)
    return arr[:, 0:2]


def load_data(f):
    pcd = o3d.io.read_point_cloud(f)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=80, std_ratio=0.5)
    sor_cloud = pcd.select_by_index(ind)

    arr = pcd2array_xy(sor_cloud)
    return arr


def get_vertex(arr, alpha=2.1, max_iter=50):
    # rcs = np.floor(arr[:]).astype(np.int32)
    # uniq, indices, counts = np.unique(rcs, axis=0, return_inverse=True, return_counts=True)
    # arr = uniq
    arr0 = arr[:, 0:2]
    arr = arr[:, 0:2]
    plt.scatter(arr[:, 0], arr[:, 1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

    # for _ in range(5):
    #     x0y0 = np.average(arr, axis=0)
    #     dists = np.sum(np.square(arr - x0y0), axis=1)
    #     averDist = np.average(dists)
    #     arr = arr[np.where(dists < averDist * np.sqrt(2.5))]
    #
    #     plt.scatter(arr[:, 0], arr[:, 1])
    #     ax = plt.gca()
    #     ax.set_aspect(1)
    #     plt.show()
    x0y0_record = np.average(arr, axis=0)

    x0y0 = np.average(arr, axis=0)
    plt.scatter(x0y0[0], x0y0[1])
    dists = np.sum(np.square(arr0 - x0y0), axis=1)
    averDist = np.average(dists)
    arr = arr0[np.where(dists < averDist * np.sqrt(alpha))]

    for i in range(max_iter - 1):
        x0y0 = np.average(arr, axis=0)
        plt.scatter(x0y0[0], x0y0[1])

        dists = np.sum(np.square(arr0 - x0y0), axis=1)
        averDist = np.average(dists)
        arr = arr0[np.where(dists < averDist * np.sqrt(alpha))]

        plt.scatter(arr[:, 0], arr[:, 1])
        ax = plt.gca()
        ax.set_aspect(1)
        plt.show()

        if (x0y0 == x0y0_record).all():
            break
        else:
            x0y0_record = x0y0
        print(i)

    x0y0 = np.average(arr, axis=0)
    dists = np.sum(np.square(arr - x0y0), axis=1)
    averDist = np.average(dists)
    arr = arr[np.where(dists > averDist * np.sqrt(alpha))]

    plt.scatter(arr[:, 0], arr[:, 1])
    ax = plt.gca()
    ax.set_aspect(1)
    plt.show()

    km = KMeans(n_clusters=4)
    km.fit(arr)
    vertices = list()
    pred = km.predict(arr)
    print(pred)
    for i in range(4):
        vertices.append(np.average(arr[np.where(pred == i)], axis=0))
    print(vertices)
    plt.scatter(arr[:, 0], arr[:, 1])
    ax = plt.gca()
    ax.set_aspect(1)
    vertices = np.array(vertices)
    plt.scatter(vertices[:, 0], vertices[:, 1], color='red')

    plt.show()
    return vertices


class CenterDetector:
    def __init__(self, points):
        self.x0y0 = np.average(points, axis=0)
        self.radius = 0
        self.points = points

    def fit(self, lr=0.1, delta_radius=1, max_iter=100):

        for i in range(max_iter):
            print(i)
            dists = np.sum(np.square(self.points - self.x0y0), axis=1)
            in_circle = np.where(dists < self.radius * self.radius)
            print('in_circle: ', in_circle)
            if in_circle[0].size > 0:
                print('in')
                delta = (self.x0y0 - np.average(self.points[in_circle], axis=0)) * lr
                dist = np.sum(np.square(delta))
                print('dist: ', dist)
                print('delta: ', delta)
                self.x0y0 = self.x0y0 + delta
                self.radius += + np.exp(-i)
            else:
                self.radius += delta_radius

            print('x0y0:', self.x0y0)
            print('radius:', self.radius)

            xy = circle(self.x0y0, self.radius, 100)
            plt.plot(xy[:, 0], xy[:, 1])

            xy = circle(self.x0y0, self.radius*np.sqrt(2), 100)
            plt.plot(xy[:, 0], xy[:, 1])

            xy = circle(self.x0y0, self.radius*np.sqrt(3), 100)
            plt.plot(xy[:, 0], xy[:, 1])

            arr_in_circle = self.points[in_circle]
            plt.scatter(self.points[:, 0], self.points[:, 1])
            plt.scatter(self.x0y0[0], self.x0y0[1], color='red')
            if in_circle[0].size > 4:
                plt.scatter(self.points[in_circle][:, 0], self.points[in_circle][:, 1], color='red')

                km = KMeans(n_clusters=4)
                km.fit(arr_in_circle)
                pred = km.predict(arr_in_circle)
                ret = np.empty(shape=[4, 2])
                for i in range(4):
                    ret[i, :] = (np.average(arr_in_circle[np.where(pred == i)], axis=0))

                plt.scatter(ret[:, 0], ret[:, 1], color='green')
            ax = plt.gca()
            ax.set_aspect(1)
            plt.show()


def circle(x0y0, radius, n):
    thetas = np.arange(0, n) * np.pi * 2 / n
    xy = np.empty(shape=[n, 2])
    xy[:, 0] = x0y0[0] + np.cos(thetas) * radius
    xy[:, 1] = x0y0[1] + np.sin(thetas) * radius
    return xy


if __name__ == '__main__':
    arr = load_data('./pcd_3.pcd')
    # arr = load_data('./lower_pcd_b3.pcd')
    # arr = load_data('./lower_pcd_b3_.pcd')

    detector = CenterDetector(arr)
    detector.fit()

    # vertices = get_vertex(arr, alpha=2, max_iter=50)
    # plt.scatter(arr[:, 0], arr[:, 1])
    # ax = plt.gca()
    # ax.set_aspect(1)

    # plt.scatter(vertices[:, 0], vertices[:, 1], color='red')

    plt.show()
