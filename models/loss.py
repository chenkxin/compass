import torch
import torch.nn as nn
import numpy as np

from pytorch3d.transforms import random_rotations
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms import quaternion_to_matrix

# using 3D rotation matrix
class ThetaBorisovLoss(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device
        self.eps = 1e-7

    def forward(self, tensor_mat_a, tensor_mat_b):

        """
         Compute difference between rotation matrix as in http://boris-belousov.net/2016/12/01/quat-dist/ on batch tensor
         :param tensor_mat_a: a tensor of rotation matrices in format [B x 3 X 3]
         :param tensor_mat_b: a tensor of rotation matrices in format [B x 3 X 3]
         :return: B values in range [0, 3.14]
         """

        mat_rotation = torch.bmm(tensor_mat_a, tensor_mat_b.transpose(2, 1))
        identity = torch.eye(3, requires_grad=True, device=self.device)

        identity = identity.reshape((1, 3, 3))
        batch_identity = identity.repeat(tensor_mat_a.size(0), 1, 1)

        trace = ((batch_identity * mat_rotation).sum(dim=(1, 2)) - 1) * 0.5

        trace = trace.clamp(min=-1 + self.eps, max=1 - self.eps)
        angles = torch.acos(trace)

        return angles

# using quaternion
class PoseDiffLoss(nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device
        self.eps = 1e-7

    def forward(self, tensor_matrix_a, tensor_matrix_b):
        """
         Compute difference between quaternion as in http://boris-belousov.net/2016/12/01/quat-dist/ on batch tensor
         :param tensor_mat_a: a tensor of rotation matrix in format [B x 3 X 3]
         :param tensor_mat_b: a tensor of rotation matrix in format [B x 3 X 3]
         :return: B values in range [0, 3.14]
         """
        tensor_quat_a = matrix_to_quaternion(tensor_matrix_a)
        tensor_quat_b = matrix_to_quaternion(tensor_matrix_b)
        temp = torch.clamp(torch.abs((tensor_quat_a *  tensor_quat_b).sum(dim=-1)), max=0.9999)
        # distance = 2 * torch.acos(temp) / np.pi
        # return distance.mean()
        distance = 2 * torch.acos(temp)
        return distance

class ChamferLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, points_src, points_trg):
        """
        Compute the Chamfer distances between two points set
        :param points_src: source input points [B X NUM_POINTS_ X CHANNELS]
        :param points_trg: target input points [B X NUM_POINTS_ X CHANNELS]
        :return two tensors, one for each set, containing the minimum squared euclidean distance between a point
        and its closest point in the other set
        """

        x, y = points_src, points_trg
        bs, num_points, points_dim = x.size()

        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))

        diag_indices = torch.arange(0, num_points).type(torch.cuda.LongTensor) if points_src.device.type == 'cuda' else torch.arange(0, num_points).type(torch.LongTensor)

        x_squared = xx[:, diag_indices, diag_indices].unsqueeze(1).expand_as(xx)
        y_squared = yy[:, diag_indices, diag_indices].unsqueeze(1).expand_as(yy)

        distances = (x_squared.transpose(2, 1) + y_squared - 2 * zz)

        return distances.min(1)[0], distances.min(2)[0]

if __name__ == '__main__':
    matrices_a = [
        [[0.6755, -0.1136, 0.7286],
        [0.5045, -0.6494, -0.5690],
        [0.5378, 0.7519, -0.3814]],
        [[-0.89565879, 0.34191775, 0.28440742],
        [0.08167378, 0.75506646, -0.65054134],
        [-0.43717814, -0.55943445, -0.70420762]],
        [[0.7789, 0.5514, -0.2988],
        [-0.5807, 0.4542, -0.6756],
        [-0.2368, 0.6998, 0.6740]]
    ]
    matrices_b = [
        [[0.9149, -0.1865, -0.3581],
         [0.3634, 0.7669, 0.5290],
         [0.1760, -0.6141, 0.7694]],
        [[-0.9399, 0.2979, -0.1667],
         [0.3361, 0.7218, -0.6050],
         [0.0599, 0.6247, 0.7786]],
        [[0.8067, -0.4157, -0.4200],
         [-0.2514, 0.4017, -0.8806],
         [-0.5348, -0.8160, -0.2196]]
    ]
    matrices_a = random_rotations(3, device='cuda')
    matrices_b = random_rotations(3, device='cuda')
    # matrices_a = torch.tensor(matrices_a).cuda()
    # matrices_b = torch.tensor(matrices_b).cuda()
    print(torch.det(matrices_a))
    print(torch.det(matrices_b))
    loss = ThetaBorisovLoss(device='cuda')
    angles = loss.forward(matrices_a, matrices_b)
    print(angles)

    loss = PoseDiffLoss(device='cuda')
    angles = loss.forward(matrices_a, matrices_b)
    print(angles)

    # loss = ThetaBorisovLoss(device='cuda')