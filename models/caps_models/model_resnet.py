# pylint: disable=E1101,R,C
import torch
import torch.nn as nn
import torch.nn.functional as F
from s2cnn import (
    S2Convolution,
    SO3Convolution,
    s2_equatorial_grid,
    so3_equatorial_grid,
    so3_integrate,
)

from .block import BaseModel, makeConv


class ModelResNet(BaseModel):
    def __init__(self, nclass):
        super().__init__()

        self.features = [6, 100, 100, nclass]
        self.bandwidths = [32, 22, 7]

        assert len(self.bandwidths) == len(self.features) - 1

        sequence = []
        # S2 laye
        sequence.append(makeConv(6, 100, self.bandwidths[0], self.bandwidths[1], "S2"))
        # SO3 layers
        for l in range(1, len(self.features) - 2):
            nfeature_in = self.features[l]
            nfeature_out = self.features[l + 1]
            b_in = self.bandwidths[l]
            b_out = self.bandwidths[l + 1]

            sequence.append(nn.BatchNorm3d(nfeature_in, affine=True))
            sequence.append(nn.ReLU())
            sequence.append(makeConv(nfeature_in, nfeature_out, b_in, b_out, "SO3"))

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


if __name__ == "__main__":
    x = torch.rand(4, 6, 64, 64).cuda()
    model = ModelResNet(10).cuda()
    print(model)
    # assert  model(x).shape == (4, 10)
