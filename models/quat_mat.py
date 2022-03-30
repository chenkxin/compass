import torch
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms import quaternion_to_matrix

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
     matrices = [
          [[-0.89565879, 0.34191775, 0.28440742],
           [0.08167378, 0.75506646, -0.65054134],
           [-0.43717814, -0.55943445, -0.70420762]],
          [[0.7789, 0.5514, -0.2988],
           [-0.5807, 0.4542, -0.6756],
           [-0.2368, 0.6998, 0.6740]]
     ]
     quats = [0.11563129, 0.91582379, -0.33029711, 0.19697718]

     ts_mat = torch.tensor(matrices)
     print(ts_mat.shape)
     res = matrix_to_quaternion(ts_mat)
     print(res)
     res = quaternion_to_matrix(res)
     print(res)