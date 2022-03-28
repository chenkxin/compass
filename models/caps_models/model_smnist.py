from .base_caps2cnn import ModelSphericalCaps


class SMNIST(ModelSphericalCaps):
    """
    Spherical capsule cnn for projected mnist on sphere
    """

    def __init__(
        self,
        nclass,
        n_hidden_capsules=5,
        n_capsule_dim=10,
        bw=30,
        use_residual_block=True,
    ):
        super(SMNIST, self).__init__(
            b_in=bw,
            primary=[
                (40, 10),  # (d_out, b_out) S^2 conv block or residual block
                (5, 10, 6),  # (n_out_caps, d_out_caps, b_out)
            ],
            hidden=[
                (5, 10, 6),
                (5, 10, 4),
                # (5, 10, 4),
                (nclass, 16, 2),
            ],
            d_in=1,
            use_residual_block=use_residual_block,
        )
