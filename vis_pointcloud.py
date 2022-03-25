import numpy as np
import open3d
from open3d import *
def visualize(p1, p2, p3):
    # from open3d.open3d.geometry import PointCloud
    # from open3d.open3d.utility import Vector3dVector
    # from open3d.open3d.visualization import draw_geometries

    point_cloud_1 = PointCloud()
    point_cloud_1.points = Vector3dVector(p1[:, 0:3].reshape(-1, 3))
    point_cloud_1.paint_uniform_color([1, 0, 0])

    point_cloud_2 = PointCloud()
    point_cloud_2.points = Vector3dVector(p2[:, 0:3].reshape(-1, 3))
    point_cloud_2.paint_uniform_color([0, 1, 0])

    point_cloud_3 = PointCloud()
    point_cloud_3.points = Vector3dVector(p3[:, 0:3].reshape(-1, 3))
    point_cloud_3.paint_uniform_color([0, 0, 1])

    draw_geometries([point_cloud_1, point_cloud_2, point_cloud_3], width=800, height=600)
    draw_geometries([point_cloud_1], width=800, height=600)
    draw_geometries([point_cloud_2], width=800, height=600)
    draw_geometries([point_cloud_3], width=800, height=600)

if __name__ == '__main__':
    points1 = np.loadtxt('original_pc_1.txt')
    points2 = np.loadtxt('AR_pc_1.txt')
    points3 = np.loadtxt('NR_pc_use_compass_1.txt')
    visualize(points1, points2, points3)

    points1 = np.loadtxt('original_pc_11.txt')
    points2 = np.loadtxt('AR_pc_11.txt')
    points3 = np.loadtxt('NR_pc_use_compass_11.txt')
    visualize(points1, points2, points3)