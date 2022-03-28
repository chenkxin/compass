import sys

import numpy as np
import torch

from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

from so3_transformer.SO3_transformer_modules import SNormSO3, get_basis_and_r, SSO3Res
from so3_transformer.equivariant_attention.fibers import Fiber

__all__ = ["SO3Transformer"]
class SO3Transformer(nn.Module):
    """SO(3) equivariant SCN with attention"""
    """
    input:
        tensor with input vectors features [Batch_size, num_vectors, vectors_features_in]
        
    Args:
        input_feature_size 4
        num_channels 32
        num_degrees 4
        div 4
        
    Returns:
        tensor with new vectors features [Batch_size, num_vectors, vectors_features_out]
    """

    def __init__(self, batch_size: int, nclass: int , input_feature_size: int, num_channels: int, d_out: int, num_degrees: int=4, div: float=4, **kwargs):
        super().__init__()
        # Build the network
        self.input_feature_size = input_feature_size
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.div = div
        self.d_out = d_out
        self.batch_size = batch_size
        self.nclass = nclass
        self.fibers = {'in': Fiber(1, input_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(1, num_degrees*self.num_channels*2)}

        blocks = self._build_gcn(self.fibers)
        self.Sblock, self.FCblock = blocks
        # initialization FCblock weight
        nn.init.kaiming_uniform_(self.FCblock[0].weight)
        nn.init.kaiming_uniform_(self.FCblock[3].weight)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0., std=1.)



    def _build_gcn(self, fibers):
        # Equivariant SO(3) transformer layers
        Sblock = []
        fin = fibers['in']
        Sblock.append(SSO3Res(fin, fibers['mid'], input_feature_size = self.input_feature_size, div=self.div, batch_size = self.batch_size, nclass = self.nclass))
        # FC layers
        FCblock = []
        FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        FCblock.append(nn.LayerNorm(self.fibers['out'].n_features))
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.fibers['out'].n_features, self.d_out))

        return nn.ModuleList(Sblock), nn.ModuleList(FCblock)

    def forward(self, x):
        S = {}
        S['d'] = x
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(S, self.num_degrees-1)

        # encoder (equivariant layers)
        h = S
        for layer in self.Sblock:
            h = layer(h, S=S, r=r, basis=basis)

        for layer in self.FCblock:
            h = layer(h)

        return h
if __name__ == '__main__':
    x = torch.randn(16,40,10) # B * N * L
    m = SO3Transformer(16,40,10,64,16) # batch_size nclass in_caps_len mid_caps_len out_caps_len
    n = m(x)
    print(n)
    print(n.shape)

