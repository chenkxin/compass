# -*- coding: utf-8 -*-
import torch

from .base_caps2cnn import ModelSphericalCaps



class ModelCaps(ModelSphericalCaps):
    """A caps model corresponding to baseline"""

    def __init__(
        self,
        nclass,
        n_hidden_capsules=5,
        n_capsule_dim=16,
        bw=32,
        use_residual_block=True,
        recon=True,
        routing="average",
        batch_size=8,
    ):
        super(ModelCaps, self).__init__(
            d_in=6,
            b_in=bw,
            nclass=nclass,
            primary=[
                (100, 8),  # (d_out, b_out) S^2 conv block or residual block
                (nclass*10, 4),
                (nclass, 10, 3),  # (nclass, 10, 4) (n_out_caps, d_out_caps, b_out)
            ],
            hidden=[
                (nclass, 10, 3),
                (nclass, n_capsule_dim, 3),
            ],
            use_residual_block=use_residual_block,
            recon=recon,
            routing=routing,
            batch_size = batch_size,
        )
