import numpy
import torch
from scipy.spatial.transform import Rotation as R

def Mat_to_Quat(matrix):
     '''
     matrix: input a 3D rotaion matrix
     return: correspond quaternion
     '''
     r = R.from_matrix(matrix)
     return r.as_quat()

def Quat_to_Mat(quat):
     '''
     quat: input a quaternion
     return: correspond 3D rotation matrix
     '''
     q = R.from_quat(quat)
     return q.as_matrix()

# TODO: dual quaternion and SE(3) matrix mutual conversion

if __name__ == '__main__':
     matrix = [[-0.89565879, 0.34191775, 0.28440742],
               [ 0.08167378, 0.75506646, -0.65054134],
               [-0.43717814, -0.55943445, -0.70420762]]
     quat = [0.11563129, 0.91582379, -0.33029711, 0.19697718]
     r = R.from_matrix(matrix)
     q = r.as_quat()
     print(q)
     Q = R.from_quat(quat)
     print(Q)
     print(Q.as_matrix())
     print(Quat_to_Mat(quat))
     print(Mat_to_Quat(matrix))
