# pylint: disable=E1101,R,C
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .block import BaseModel
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid
from s2cnn import (
    S2Convolution,
    SO3Convolution,
    s2_equatorial_grid,
    so3_equatorial_grid,
    so3_integrate,
)


class ModelBaseline_3d(BaseModel):
    """
    Designed for 3d object
    """

    def __init__(self, nclass):
        super().__init__()

        self.features = [6, 100, 100, nclass]
        self.bandwidths = [32, 22, 7]

        assert len(self.bandwidths) == len(self.features) - 1

        sequence = []

        # S2 layer
        grid = s2_equatorial_grid(
            max_beta=0, n_alpha=2 * self.bandwidths[0], n_beta=1
        )  # grid is a tuple 128*128
        sequence.append(
            S2Convolution(
                self.features[0],
                self.features[1],
                self.bandwidths[0],
                self.bandwidths[1],
                grid,
            )
        )

        # SO3 layers
        for l in range(1, len(self.features) - 2):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l + 1]
            b_in = self.bandwidths[l]
            b_out = self.bandwidths[l + 1]

            sequence.append(nn.BatchNorm3d(nfeature_in, affine=True))
            sequence.append(nn.ReLU())
            grid = so3_equatorial_grid(
                max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1
            )
            sequence.append(
                SO3Convolution(nfeature_in, nfeature_out, b_in, b_out, grid)
            )

        sequence.append(nn.BatchNorm3d(self.features[-2], affine=True))
        sequence.append(nn.ReLU())

        self.sequential = nn.Sequential(*sequence)

        # Output layer
        output_features = self.features[-2]
        self.out_layer = nn.Linear(output_features, self.features[-1])

    def forward(self, x):  # pylint: disable=W0221
        x = self.sequential(x)  # [batch, feature, beta, alpha, gamma]
        x = so3_integrate(x)  # [batch, feature]

        x = self.out_layer(x)
        return F.log_softmax(x, dim=1)


class ModelBaseline_SMNIST(nn.Module):
    def __init__(self):
        super(ModelBaseline_SMNIST, self).__init__()

        f1 = 20
        f2 = 40
        f_output = 10

        b_in = 30
        b_l1 = 10
        b_l2 = 6

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.conv1 = S2Convolution(
            nfeature_in=1, nfeature_out=f1, b_in=b_in, b_out=b_l1, grid=grid_s2
        )

        self.conv2 = SO3Convolution(
            nfeature_in=f1, nfeature_out=f2, b_in=b_l1, b_out=b_l2, grid=grid_so3
        )

        self.out_layer = nn.Linear(f2, f_output)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = so3_integrate(x)

        x = self.out_layer(x)

        return x


class ModelBaseline_SMNIST_Deep(nn.Module):
    def __init__(self, bandwidth=30):
        super(ModelBaseline_SMNIST_Deep, self).__init__()

        grid_s2 = s2_near_identity_grid(n_alpha=6, max_beta=np.pi / 16, n_beta=1)
        grid_so3_1 = so3_near_identity_grid(
            n_alpha=6, max_beta=np.pi / 16, n_beta=1, max_gamma=2 * np.pi, n_gamma=6
        )
        grid_so3_2 = so3_near_identity_grid(
            n_alpha=6, max_beta=np.pi / 8, n_beta=1, max_gamma=2 * np.pi, n_gamma=6
        )
        grid_so3_3 = so3_near_identity_grid(
            n_alpha=6, max_beta=np.pi / 4, n_beta=1, max_gamma=2 * np.pi, n_gamma=6
        )
        grid_so3_4 = so3_near_identity_grid(
            n_alpha=6, max_beta=np.pi / 2, n_beta=1, max_gamma=2 * np.pi, n_gamma=6
        )

        self.convolutional = nn.Sequential(
            S2Convolution(
                nfeature_in=1,
                nfeature_out=8,
                b_in=bandwidth,
                b_out=bandwidth,
                grid=grid_s2,
            ),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in=8,
                nfeature_out=16,
                b_in=bandwidth,
                b_out=bandwidth // 2,
                grid=grid_so3_1,
            ),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in=16,
                nfeature_out=16,
                b_in=bandwidth // 2,
                b_out=bandwidth // 2,
                grid=grid_so3_2,
            ),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in=16,
                nfeature_out=24,
                b_in=bandwidth // 2,
                b_out=bandwidth // 4,
                grid=grid_so3_2,
            ),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in=24,
                nfeature_out=24,
                b_in=bandwidth // 4,
                b_out=bandwidth // 4,
                grid=grid_so3_3,
            ),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in=24,
                nfeature_out=32,
                b_in=bandwidth // 4,
                b_out=bandwidth // 8,
                grid=grid_so3_3,
            ),
            nn.ReLU(inplace=False),
            SO3Convolution(
                nfeature_in=32,
                nfeature_out=64,
                b_in=bandwidth // 8,
                b_out=bandwidth // 8,
                grid=grid_so3_4,
            ),
            nn.ReLU(inplace=False),
        )

        self.linear = nn.Sequential(
            # linear 1
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(inplace=False),
            # linear 2
            nn.BatchNorm1d(64),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(inplace=False),
            # linear 3
            nn.BatchNorm1d(32),
            nn.Linear(in_features=32, out_features=10),
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = so3_integrate(x)
        x = self.linear(x)
        return x
