import numpy as np
import math

import torch
import torch.nn as nn

from models import soft_argmax as sfa
from utils import sb_torch as uto

from s2cnn import so3_equatorial_grid, SO3Convolution, so3_near_identity_grid

from models.caps_models import block
from models.caps_models.so3_transformer import SO3Transformer
from models.cholesky import Cholesky
class CapsuleLayer(nn.Module):
    def __init__(self, bandwidths, features, capsules, use_residual_block=True):
        super().__init__()

        self.bandwidths = bandwidths # It is a list contained in_band_width and out_band_width for every capsule layer
        self.features = features # It is a list contained in_features and out_features for every capsule layer
        # self.softmax_temp = softmax_temp
        self.capsules = capsules # It is a list contained in_capsules_nums and out_capsules_nums for every capsule layer
        self.use_residual_block = use_residual_block

        caps_sequence = []
        # 胶囊层
        self.features = [40, 40, 40, 40]
        self.bandwidths = [24, 24, 24, 24]
        self.capsules = [0, 4, 4, 4] # 初始胶囊层没有输入胶囊个数
        # 初级胶囊层
        l = 0
        caps_sequence.append(block.PrimaryCapsuleLayer(
            in_features=self.features[l],
            num_out_capsules=self.capsules[l + 1],
            capsule_dim=int(self.features[l + 1] / self.capsules[l + 1]),
            b_in=self.bandwidths[l],
            b_out=self.bandwidths[l + 1],
            use_residual_block=self.use_residual_block
        ))
        l = l + 1
        # routing or so3_transformer
        caps_sequence.append(block.ConvolutionalCapsuleLayer(
            num_in_capsules=self.capsules[l],
            in_capsule_dim=int(self.features[l] / self.capsules[l]),
            num_out_capsules=self.capsules[l + 1],
            out_capsule_dim=int(self.features[l + 1] / self.capsules[l + 1]),
            b_in=self.bandwidths[l],
            b_out=self.bandwidths[l + 1],
            # num_channels = 64,
            is_class=False,
            use_residual_block=True,
            routing="average",
            batch_size=8,
            nclass=10
        ))
        l = l + 1
        caps_sequence.append(block.RotationEstimateLayer(
            num_in_capsules=self.capsules[l],
            in_capsule_dim=int(self.features[l] / self.capsules[l]),
            b_in=self.bandwidths[l],
            b_out=self.bandwidths[l + 1],
        ))

        # self.caps_layer = nn.Sequential(*caps_sequence)

        # self.soft_argmarx = sfa.SoftArgmax3D(0.0, 1.0, 'Parzen', float(self.bandwidths[-1] * 2.0), self.softmax_temp)

        # use cholesky to orthonorm 3D rotation matrix
        # self.cholesky = Cholesky()

    def forward(self, input):  # pylint: disable=W0221
    #     lrf_features_map = self.caps_layer(input)
    #
    #     arg_maxima = self.soft_argmarx(lrf_features_map)
    #
    #     size_alphas = lrf_features_map.shape[-2]
    #     size_betas = lrf_features_map.shape[-1]
    #     size_gammas = lrf_features_map.shape[-3]

        # Swap Alpha and Beta
        # arg_maxima = arg_maxima.reshape(-1, 3)

        # alphas = math.pi * arg_maxima[:, 1] / (size_alphas * 0.5)
        # betas = math.pi * (2 * arg_maxima[:, 0] + 1) / (4 * (size_betas * 0.5))
        # gammas = math.pi * arg_maxima[:, 2] / (size_gammas * 0.5)

        # mat_lrf = uto.b_get_rotation_matrices_from_euler_angles_on_tensor(alphas, betas, gammas, device=input.device)

        # cholesky
        # mat_lrf[0] = self.cholesky(mat_lrf[0])
        # return lrf_features_map, mat_lrf
        pass

    def __repr__(self):
        layer_str = ""
        for name, param in self.named_parameters():
            if 'kernel' in name:
                layer_str += "Name: {} - Shape {}".format(name, param.transpose(2, 1).shape) + "\n"

        return super(CapsuleLayer, self).__repr__() + layer_str

if __name__ == '__main__':
    a = torch.randn(1, 40, 48, 48, 48)
    print(a.shape)
    # bandwidths, features, capsules, use_residual_block = True
    # self.features = [40, 40, 40, 40]
    # self.bandwidths = [24, 24, 24, 24]
    # self.capsules = [0, 4, 4, 4] # 初始胶囊层没有输入胶囊个数

    layer = CapsuleLayer(bandwidths=[24, 24, 24, 24], features=[40, 40, 40, 40], capsules=[0, 4, 4, 4], use_residual_block=True)
    res = layer(a)
    print(res)
    print(res[0].shape)
    print(res[1].shape)
