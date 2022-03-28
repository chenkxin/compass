import torch
from torch import nn

from .base_caps2cnn import ModelSphericalCaps
from .block import BaseModel


class MSVC(BaseModel):
    def __init__(
        self,
        nclass,
        bandwidths,
        use_residual_block=True,
        n_hidden_capsules=5,
        n_capsule_dim=16,
        recon=True,
        **kwargs
    ):
        """

        Args:
            nclass:
            bandwidths:
            use_residual_block:
            **kwargs:

        Example:
            model = MSVC(nclass=10, bandwidths=[32,16,8])
            x = [torch.rand([4, 6, 64, 64]),
                 torch.rand([4, 6, 32, 32]),
                 torch.rand([4, 6, 16, 16])
                 ]
            assert model(x).shape == (4, 10)
        """
        #       Input  S2conv         InitCaps           CapsConv             SO(3)conv
        # b: (d=6,b=32) --> (d=20,b=7) --> (n=5,d=16,b=7) --> (nclass, 16, 7) --> (nclass, 16)
        super(MSVC, self).__init__()
        network1 = ModelSphericalCaps(
            n_capsule_dim=n_capsule_dim,
            nclass=nclass,
            d_in=6,
            b_in=bandwidths[0],
            primary=[
                (20, 7),  # (d_out, b_out) S^2 conv block or residual block
                (
                    n_hidden_capsules,
                    n_capsule_dim,
                    7,
                ),  # (n_out_caps, d_out_caps, b_out)
            ],
            hidden=[(nclass, n_capsule_dim, 7)],
            use_residual_block=use_residual_block,
            recon=recon,
        )
        #       Input  S2conv         InitCaps           CapsConv             SO(3)conv
        #    (d=6,b=16) --> (d=50,b=16) --> (n=5,d=16,b=7) --> (nclass, 16, 7) --> (nclass, 16)
        network2 = ModelSphericalCaps(
            n_capsule_dim=n_capsule_dim,
            nclass=nclass,
            d_in=6,
            b_in=bandwidths[1],
            primary=[
                (20, 7),  # (d_out, b_out) S^2 conv block or residual block
                (
                    n_hidden_capsules,
                    n_capsule_dim,
                    7,
                ),  # (n_out_caps, d_out_caps, b_out)
            ],
            hidden=[(nclass, n_capsule_dim, 7)],
            use_residual_block=use_residual_block,
            recon=recon,
        )
        #       Input  S2conv        InitCaps           CapsConv           SO(3)conv
        #    (d=6,b=8) --> (d=50,b=8) --> (n=5,d=16,b=7) --> (n=5,d=16,b=7) --> (nclass, 16)
        network3 = ModelSphericalCaps(
            n_capsule_dim=n_capsule_dim,
            nclass=nclass,
            d_in=6,
            b_in=bandwidths[2],
            primary=[
                (20, 7),  # (d_out, b_out) S^2 conv block or residual block
                (
                    n_hidden_capsules,
                    n_capsule_dim,
                    7,
                ),  # (n_out_caps, d_out_caps, b_out)
            ],
            hidden=[(nclass, n_capsule_dim, 7)],
            use_residual_block=use_residual_block,
            recon=recon,
        )

        self.networks = nn.ModuleList([network1, network2, network3])

        activation = nn.ReLU
        # self.fc = nn.Sequential(
        #    nn.Linear(nclass * 3, nclass),
        #    nn.Softmax(dim=-1)
        # )
        # self.fc = nn.Sequential(
        #     nn.Linear(nclass * 3, 256),
        #     activation(),
        #     nn.Linear(256, 512),
        #     activation(),
        #     nn.Linear(512, 256),
        #     activation(),
        #     nn.Linear(256, nclass),
        #     nn.Softmax(dim=-1)
        # )
        self.fc = nn.Sequential(
            nn.Linear(nclass * 3, 256),
            activation(),
            nn.Linear(256, nclass),
            nn.Softmax(dim=-1),
        )
        self.middle_class_capsule = None
        self.recon = recon

    def forward(self, inputs, target=None):
        """
        Two strategies:
            1: compute capsule lengths of each network, concate them and
            use mlp

            2: use batch mlp (each capsule one mlp)
        Args:
            inputs:

        Returns:
            Final class capsule
        """

        out = []
        if self.recon:
            x_recons = []
            for n, x in zip(self.networks, inputs):
                embedding, y, recon, nclass = n(x, target)
                out.append(embedding)  # [(B,nclass), ...]
                x_recons.append(recon)
            self.middle_class_capsule = out
            out = torch.cat(out, dim=1)  # (B,nclass*n_networks) where n_networks=3
            out = self.fc(out)  # (B,nclass)
            return out, y, x_recons, nclass
        else:
            for n, x in zip(self.networks, inputs):
                embedding = n(x, target)
                out.append(embedding)  # [(B,nclass), ...]
            self.middle_class_capsule = out
            out = torch.cat(out, dim=1)  # (B,nclass*n_networks) where n_networks=3
            out = self.fc(out)  # (B,nclass)
            return out


class MSVCCaps(MSVC):
    def __init__(self, *args, **kwargs):
        super(MSVCCaps, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        super().forward(*args, **kwargs)
        return self.middle_class_capsule
