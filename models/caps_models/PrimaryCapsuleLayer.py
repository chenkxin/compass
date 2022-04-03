from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from models.caps_models.functional import degree_score, squash
from models.caps_models.so3_transformer import SO3Transformer
from s2cnn import (
    S2Convolution,
    SO3Convolution,
    s2_equatorial_grid,
    so3_equatorial_grid,
    so3_integrate,
)
class PrimaryCapsuleLayer(nn.Module):
    """
    Convert a SO(3) feature to capsules, we first conv in_features to
        out_features = num_out_capsules*capsules,
    then reshape to capsules
    """

    def __init__(
        self,
        in_features=100,
        num_out_capsules=10,
        capsule_dim=15,
        b_in=32,
        b_out=32,
        use_residual_block=True,
    ):
        super().__init__()
        out_features = capsule_dim * num_out_capsules
        self.capsules = get_capsule_block(
            in_features,
            out_features,
            b_in,
            b_out,
            use_residual_block=use_residual_block,
        )
        self.num_out_capsules = num_out_capsules
        self.capsule_dim = capsule_dim
        self.b_out = b_out

    def forward(self, x):
        """
        Args:
            x (N, in_features, 2*b_in, 2*b_in, 2*b_in)
        Returns:
            result (N, num_out_capsules, capsule_dim, 2*b_out, 2*b_out, 2*b_out)
        """
        N = x.shape[0]
        x = self.capsules(x)  # (N, out_features, 2*b_out, 2*b_out, 2*b_out)
        x = x.view(
            -1,
            self.num_out_capsules,
            self.capsule_dim,
            2 * self.b_out,
            2 * self.b_out,
            2 * self.b_out,
        )
        x = squash(x, dim=2)
        return x