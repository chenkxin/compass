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
class LrfLayer(nn.Module):
    def __init__(self, bandwidths, features, softmax_temp, use_equatorial_grid, use_residual_block=True):
        super().__init__()

        self.bandwidths = bandwidths
        self.features = features
        self.softmax_temp = softmax_temp
        self.use_residual_block = use_residual_block

        lrf_sequence = []
        # SO3 layers
        for l in range(0, len(self.features) - 1):
            num_feature_in = self.features[l]
            num_feature_out = self.features[l + 1]

            bw_in = self.bandwidths[l]
            bw_out = self.bandwidths[l + 1]

            lrf_sequence.append(nn.BatchNorm3d(num_feature_in, affine=True))
            lrf_sequence.append(nn.ReLU())

            grid = so3_equatorial_grid(max_beta=0, max_gamma=0, n_alpha=2 * bw_in, n_beta=1, n_gamma=1) if use_equatorial_grid else so3_near_identity_grid(max_beta=np.pi / 8, max_gamma=2*np.pi, n_alpha=8, n_beta=3, n_gamma=8)
            lrf_sequence.append(SO3Convolution(num_feature_in, num_feature_out, bw_in, bw_out, grid))
        lrf_sequence.append(nn.BatchNorm3d(self.features[-1], affine=True))

        # 胶囊层
        # 初级胶囊层
        lrf_sequence.append(block.PrimaryCapsuleLayer(
            in_features=self.features[-1], # 40
            num_out_capsules=4,
            capsule_dim=10,
            b_in=24,
            b_out=24,
            use_residual_block=self.use_residual_block
        ))
        # routing or so3_transformer
        lrf_sequence.append(block.ConvolutionalCapsuleLayer(
            num_in_capsules=4,
            in_capsule_dim=10,
            num_out_capsules=4,
            out_capsule_dim=10,
            b_in=24,
            b_out=24,
            # num_channels = 64,
            is_class=False,
            use_residual_block=True,
            routing="average",
            batch_size=8,
            nclass=10
        ))
        lrf_sequence.append(block.RotationEstimateLayer(
            num_in_capsules=4,
            in_capsule_dim=10,
            b_in=24,
            b_out=24,
        ))

        self.lrf_layer = nn.Sequential(*lrf_sequence)

        self.soft_argmarx = sfa.SoftArgmax3D(0.0, 1.0, 'Parzen', float(self.bandwidths[-1] * 2.0), self.softmax_temp)

        # use cholesky to orthonorm 3D rotation matrix
        self.cholesky = Cholesky()

    def forward(self, input):  # pylint: disable=W0221
        lrf_features_map = self.lrf_layer(input)

        arg_maxima = self.soft_argmarx(lrf_features_map)

        size_alphas = lrf_features_map.shape[-2]
        size_betas = lrf_features_map.shape[-1]
        size_gammas = lrf_features_map.shape[-3]

        # Swap Alpha and Beta
        arg_maxima = arg_maxima.reshape(-1, 3)

        alphas = math.pi * arg_maxima[:, 1] / (size_alphas * 0.5)
        betas = math.pi * (2 * arg_maxima[:, 0] + 1) / (4 * (size_betas * 0.5))
        gammas = math.pi * arg_maxima[:, 2] / (size_gammas * 0.5)

        mat_lrf = uto.b_get_rotation_matrices_from_euler_angles_on_tensor(alphas, betas, gammas, device=input.device)

        # cholesky
        mat_lrf[0] = self.cholesky(mat_lrf[0])
        return lrf_features_map, mat_lrf

    def __repr__(self):
        layer_str = ""
        for name, param in self.named_parameters():
            if 'kernel' in name:
                layer_str += "Name: {} - Shape {}".format(name, param.transpose(2, 1).shape) + "\n"

        return super(LrfLayer, self).__repr__() + layer_str

if __name__ == '__main__':
    a = torch.randn(1, 4, 48, 48, 48)
    print(a.shape)
    # bandwidths, features, softmax_temp, use_equatorial_grid
    layer = LrfLayer([24, 24, 24], [4, 50, 40], 1.0, True, True)
    res = layer(a)
    print(res)
    print(res[0].shape)
    print(res[1].shape)
