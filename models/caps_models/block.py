# -*- coding: utf-8 -*-
"""
Basic blocks for capsule network, we wrote some high level api:
 - get_inital_block(in_features, out_features, b_in, b_out, use_residual_block=True)
 - get_capsule_block(in_features, out_features, b_in, b_out, use_residual_block=True)
 - `PrimaryCapsuleLayer` to conver a SO(3) feature to capsules
 - `ConvolutionalCapsuleLayer` using degree routing to compute capsules next layer
"""
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
from lib.functional import degree_score, squash
from so3_transformer import SO3Transformer
from s2cnn import (
    S2Convolution,
    SO3Convolution,
    s2_equatorial_grid,
    so3_equatorial_grid,
    so3_integrate,
)


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic
        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


def makeConv(in_features=6, out_features=100, b_in=32, b_out=32, type="S2"):
    # S2 or SO3 conv
    if type == "S2":
        grid = s2_equatorial_grid(max_beta=0, n_alpha=2 * b_in, n_beta=1)
        return S2Convolution(in_features, out_features, b_in, b_out, grid)
    elif type == "SO3":
        grid = so3_equatorial_grid(
            max_beta=0, max_gamma=0, n_alpha=2 * b_in, n_beta=1, n_gamma=1
        )
        return SO3Convolution(in_features, out_features, b_in, b_out, grid)


class Block(nn.Module):
    """
    A basic residual block for sphere convolution, DONT USE DIRECTLY
    """

    def __init__(self, in_features, out_features, b_in, b_out, conv1="S2"):
        super().__init__()
        # S2 and SO3
        self.road1 = nn.Sequential(
            makeConv(in_features, out_features, b_in, b_out * 2, conv1),
            nn.BatchNorm3d(out_features, affine=True),
            nn.SELU(),
            makeConv(out_features, out_features, b_out * 2, b_out, "SO3"),
            nn.BatchNorm3d(out_features, affine=True),
            nn.SELU(),
        )
        # short cut
        self.road2 = nn.Sequential(
            makeConv(in_features, out_features, b_in, b_out, conv1),
            nn.BatchNorm3d(out_features, affine=True),
            nn.SELU(),
        )
        self.road3 = nn.Sequential(nn.BatchNorm3d(out_features, affine=True), nn.SELU())

    def forward(self, x):
        """
        Args:
            x (N, in_features, 2*b_in, 2*b_in, 2*b_in)
        Returns:
            result (N, out_features, 2*b_out, 2*b_out, 2*b_out)
        """
        x1 = self.road1(x)
        x2 = self.road2(x)
        result = self.road3(x1 + x2)  # no bn for output
        return result


class InitialResidualBlock(Block):
    def __init__(self, in_features, out_features, b_in, b_out):
        super().__init__(in_features, out_features, b_in, b_out, conv1="S2")


class CapsuleResidualBlock(Block):
    def __init__(self, in_features, out_features, b_in, b_out):
        super().__init__(in_features, out_features, b_in, b_out, conv1="SO3")


# High level api
def get_inital_block(in_features, out_features, b_in, b_out, use_residual_block=True):
    # used in primary capsule layer, only maps features from low to high dimension
    # input: `in_features` with `b_in` (BS, in_features, b_in, b_in)
    # output: `out_features` with `b_out`(BS, out_features, b_out, b_out)
    if use_residual_block:
        return InitialResidualBlock(in_features, out_features, b_in, b_out)
    else:
        return makeConv(in_features, out_features, b_in, b_out, "S2")


def get_capsule_block(in_features, out_features, b_in, b_out, use_residual_block=True):
    # used in capsule layer, only maps features from one to another dimension
    if use_residual_block:
        return CapsuleResidualBlock(in_features, out_features, b_in, b_out)
    else:
        return makeConv(in_features, out_features, b_in, b_out, "SO3")


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


class CapsulePredictionLayer(nn.Module):
    """Given capsules u, compute u_hat"""

    def __init__(
        self,
        in_features=100,
        num_out_capsules=10,
        out_capsule_dim=15,
        b_in=32,
        b_out=32,
        use_residual_block=True,
    ):
        super().__init__()
        out_features = out_capsule_dim * num_out_capsules
        self.capsules = get_capsule_block(
            in_features,
            out_features,
            b_in,
            b_out,
            use_residual_block=use_residual_block,
        )
        self.num_out_capsules = num_out_capsules
        self.out_capsule_dim = out_capsule_dim
        self.b_out = b_out

    def forward(self, x):
        """
        Args:
            x capsule (N, num_in_capsules,in_capsule_dim, 2*b_in, 2*b_in, 2*b_in)
        Returns:
            u_hat (N, num_in_capsules, num_out_capsules, out_capsule_dim, 2*b_out, 2*b_out, 2*b_out)
        """
        # 1.reshape input to (N*num_in_capsules,in_capsule_dim, 2*b_in, 2*b_in,2*b_in)
        N, num_in_capsules, in_capsule_dim, B_in, B_in1, B_in2 = x.shape
        assert B_in == B_in1 == B_in2
        x = x.view(N * num_in_capsules, in_capsule_dim, B_in, B_in, B_in)

        # 2.use residual block to map in_capsule_dim to out_capsule_dim*num_out_capsules
        # Note: this step will compute relationship between in and out capsules
        x = self.capsules(x)

        # 3.split dim0 and dim1
        b_out = self.b_out
        out_capsule_dim = self.out_capsule_dim
        num_out_capsules = self.num_out_capsules
        return x.view(
            N,
            num_in_capsules,
            num_out_capsules,
            out_capsule_dim,
            2 * b_out,
            2 * b_out,
            2 * b_out,
        )


class ConvolutionalCapsuleLayer(nn.Module):
    def __init__(
        self,
        num_in_capsules,
        in_capsule_dim,
        num_out_capsules,
        out_capsule_dim,
        b_in,
        b_out,
        num_channels=64,
        is_class=False,
        use_residual_block=True,
        routing="average",
        batch_size = 8,
        nclass = 10,
    ):
        super().__init__()
        self.routing = routing
        self.batch_size = batch_size
        self.out_capsule_dim = out_capsule_dim
        self.b_out = b_out
        self.b_in = b_in
        self.nclass = nclass
        if routing == "so3_transformer":

            self.transformer = SO3Transformer(batch_size*((2*b_in)**3), nclass, in_capsule_dim, num_channels, out_capsule_dim)

        self.prediction_network = CapsulePredictionLayer(
            in_capsule_dim,
            num_out_capsules,
            out_capsule_dim,
            b_in,
            b_out,
            use_residual_block,
        )
        self.is_class = is_class

    def forward(self, x):
        """Capsule convolution layer using degree routing
        Args:
            x capsules, (N, num_in_capsules,in_capsule_dim, 2*b_in, 2*b_in, 2*b_in)
        Returns:
            result (N, num_out_capsules,out_capsule_dim, 2*b_out, 2*b_out, 2*b_out)
        """

        if self.routing == "so3_transformer":
            if self.nclass<=40:
                if len(x.shape) > 3:
                  N, N_c, D_c, l, _, _ = x.shape
                  x = x.permute(0,3,4,5,1,2).reshape(-1, N_c, D_c) #(N*2*b_in*2*b_in*2*b_in,num_in_capsules,in_capsule_dim)
                x = self.transformer(x) #(N*2*b_in*2*b_in*2*b_in,num_in_capsules,out_capsule_dim)
                x = squash(x,dim=2)
                if self.is_class:
                    x = x.reshape(self.batch_size,2*self.b_out,2*self.b_out,2*self.b_out,self.nclass,self.out_capsule_dim).permute(0,4,5,1,2,3)
                    return so3_integrate(x)
                else:
                    return x
            else:
                if len(x.shape) > 3:
                    x = so3_integrate(x)
                    x = self.transformer(x)
                    x = squash(x, dim=2)
                else:
                    x = self.transformer(x)
                    x = squash(x, dim=2)
                return x
        else:
            # 1. predict u_hat
            # (N, num_in_capsules, num_out_capsules, out_capsule_dim, 2*b_out, 2*b_out, 2*b_out)
            u_hat = self.prediction_network(x)

            if self.routing == "average":
                # 2. compute mean
                # (N, num_out_capsules, out_capsule_dim, 2 * b_out, 2 * b_out, 2 * b_out)
                s_j = torch.mean(u_hat, dim=1, keepdim=False)

                # 3. squash
                # (N, num_out_capsules, out_capsule_dim, 2*b_out, 2*b_out, 2*b_out)
                v_j = squash(s_j, dim=2)
            elif self.routing == "degree_score":
                # 2. compute the score
                score = degree_score(u_hat)

                # 3. compute capsules of next layer
                # (N, 1, num_out_capsules, out_capsule_dim, 2*b_out, 2*b_out, 2*b_out)
                s_j = (score * u_hat).sum(dim=1, keepdim=True)
                v_j = squash(s_j, dim=3).squeeze(dim=1)
            else:
                raise Exception("No routing method named {}".format(self.routing))

            if self.is_class:
                return so3_integrate(v_j)
            else:
                return v_j
