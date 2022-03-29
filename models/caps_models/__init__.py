import glob
import os

import torch
import torch.nn as nn
from models.caps_models.utils import get_oldest_state

from .model_baseline import (
    ModelBaseline_3d,
    ModelBaseline_SMNIST,
    ModelBaseline_SMNIST_Deep,
)
from .model_caps import ModelCaps
from .model_resnet import ModelResNet
from .model_smnist import SMNIST
from .model_msvc import MSVC, MSVCCaps
from models.caps_models.utils import get_bw


def makeModel(
    name,
    model_dir,
    nclass=10,
    device=None,
    is_distributed=False,
    use_residual_block=True,
    load=False,
    config=None,
):
    recon = True if config.loss == "CapsuleRecon" else False
    if name == "caps":
        model = ModelCaps(
            nclass,
            use_residual_block=use_residual_block,
            recon=recon,
            bw=get_bw(config),
            routing=config.routing,
            batch_size=config.batch_size,
        )
    elif name == "baseline":
        model = ModelBaseline_3d(nclass)
    elif name == "resnet":
        model = ModelResNet(nclass)
    elif name == "smnist":
        model = SMNIST(nclass, use_residual_block=use_residual_block)
    elif name == "smnist_baseline":
        model = ModelBaseline_SMNIST()
    elif name == "smnist_baseline_deep":
        model = ModelBaseline_SMNIST_Deep()
    # TODO: can only use three channels
    elif name == "msvc":
        model = MSVC(
            nclass=nclass,
            bandwidths=[32, 16, 8],
            use_residual_block=use_residual_block,
            recon=recon,
        )
    elif name == "msvc_caps":
        model = MSVCCaps(
            nclass=nclass,
            bandwidths=[32, 16, 8],
            use_residual_block=use_residual_block,
            recon=recon,
        )
    else:
        raise ValueError(f"Not implemented model for {name}")

    if device:
        model = model.to(device)
        if is_distributed:
            model = nn.DataParallel(model)

    # continue training feature
    LAST_EPOCH = -1
    if load:
        state, LAST_EPOCH = get_oldest_state(model_dir, name)
        if state:
            model.load_state_dict(state["model"])
    return LAST_EPOCH, model
