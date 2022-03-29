import glob
import json
import os
import subprocess
import socket
import matplotlib.pyplot as plt
import numpy
import numpy as np
import torch
from monty.collections import AttrDict
from tqdm import tqdm


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """

    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print("INFO: Early stopping")
                self.early_stop = True


def show_recon(image, label, index, data=False):
    """
    A script for viewing spherical images of mnist
    """
    from mayavi import mlab

    def create_shpere(b=60):
        # Make sphere, choose colors
        phi, theta = np.mgrid[0 : np.pi : b * 1j, 0 : 2 * np.pi : b * 1j]
        x, y, z = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)
        return x, y, z

    mlab.figure(1, bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(1000, 800))
    # index = 3
    x, y, z = create_shpere()
    im = np.array(image.detach().cpu().numpy(), dtype=np.dtype("uint8"))
    # mlab.mesh(x, y + n * 2 + 0.2, z, scalars=im, colormap="coolwarm")
    mlab.mesh(x, y, z, scalars=im, colormap="coolwarm")
    mlab.view(0, 170, 10)
    if not data:
        filename = (
            "smnist_recon" + "_lable_" + str(label.item()) + "_" + str(index) + ".png"
        )
    else:
        filename = (
            "smnist_real" + "_lable_" + str(label.item()) + "_" + str(index) + ".png"
        )
    mlab.savefig(filename)
    # mlab.show()
    # directly on 2d
    # plt.imshow(im,cmap ='gray')
    # plt.show()


def _get_train_data(data, device, config):
    # for multi instance-one target input
    if config.model_name == "msvc":
        inputs = [i.to(device) for i in data]
        data = inputs
    else:
        data = data.to(device)
    return data


def _compute_prediction_and_loss(model, criterion, data, target, device, config):
    if config.loss == "CapsuleRecon":
        prediction, x_recon, nclass = model(data, target)
        loss = criterion(prediction, target, data, x_recon, nclass)
    elif config.loss == "MarginLoss":
        prediction = model(data)
        loss = criterion(prediction, target)
    else:
        target = target.to(device)
        prediction = model(data)
        loss = criterion(prediction, target)
    return prediction, loss


def train_step(model, optimizer, criterion, data, target, device, config):
    model.train()
    data = _get_train_data(data, device, config)
    prediction, loss = _compute_prediction_and_loss(
        model, criterion, data, target, device, config
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    correct = torch.sum(torch.argmax(prediction, dim=1).cpu() == target)
    return loss.item(), correct.item()


def test_step(model, criterion, data, target, device, config):
    model.eval()
    with torch.no_grad():
        data = _get_train_data(data, device, config)
        prediction, loss = _compute_prediction_and_loss(
            model, criterion, data, target, device, config
        )
        correct = torch.sum(torch.argmax(prediction, dim=1).cpu() == target)
    # show_recon(data.view(data.size(2),data.size(2)), label, index, data = True)
    # show_recon(x_recon.view(data.size(2),data.size(2)), pre_label, index, data = False)
    # pre = torch.topk(prediction.data, k=overlap_num)[1]
    # pre_reversed = torch.flip(pre, dims=[1])
    # result1 = pre.eq(target.data)
    # result2 = pre_reversed.eq(target.data)
    # right_bool = result1 | result2
    # correct = right_bool.long().cpu().sum()
    # return loss.item(), correct.item(), right_bool
    return loss.item(), correct.item()


def _update_acc_dict(acc_dict, target, right_bool):
    """update acc dict after every evaluating epoch"""
    if acc_dict is None:
        return
    target = target.reshape(-1)
    right_bool = right_bool.reshape(-1)
    for i, e in enumerate(target):
        if right_bool[i]:
            acc_dict[e.item()]["right"] += 1
        acc_dict[e.item()]["total"] += 1
        acc_dict[e.item()]["acc"] = round(
            acc_dict[e.item()]["right"] / acc_dict[e.item()]["total"], 2
        )
    return acc_dict


def _plot_acc(acc_for_every_cls):
    x = [acc_for_every_cls[i]["acc"] for i in range(10)]
    plt.bar(range(len(x)), x)
    plt.xticks(range(len(x)))
    plt.yticks(np.linspace(0, 1, 5, endpoint=True))
    plt.show()


def evaluate(model, criterion, testloader, device, config, **kwargs):
    """

    Args:
        model:
        criterion:
        testloader:
        device:
        **kwargs:

    Returns:

    """
    if "nclass" in kwargs:
        nclases = kwargs["nclass"]
        acc_for_every_cls = dict()
        for i in range(nclases):
            acc_for_every_cls[i] = {"right": 0, "total": 0, "acc": 0}
    else:
        acc_for_every_cls = None
    # for Test
    total_correct = 0
    total_loss = 0
    N = 0

    for batch_idx, (data, target) in enumerate(testloader):
        B = target.shape[0]
        # target = target.reshape(B, -1)
        # B, overlap_num = target.shape
        # l, correct, right_bool = test_step(model, criterion, data, target, device)
        l, correct = test_step(model, criterion, data, target, device, config)
        # acc_for_every_cls = _update_acc_dict(acc_for_every_cls, target, right_bool)
        total_loss += l
        total_correct += correct

        # N += B * overlap_num
        N += B

    # if "plot" in kwargs and kwargs["plot"]:
    # _plot_acc(acc_for_every_cls)
    acc = total_correct / N
    return AttrDict(loss=total_loss, acc=acc, acc_for_every_cls=acc_for_every_cls)


def get_learning_rate(epoch, learning_rate):
    limits = [100, 200]
    lrs = [1, 0.1, 0.01]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate


def update_learning_rate(optimizer, epoch, learning_rate):
    lr = get_learning_rate(epoch, learning_rate)
    for p in optimizer.param_groups:
        p["learning_rate"] = lr


def check_dir(expr_name):
    model_dir = os.path.join("logs", expr_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs(os.path.join(model_dir, "tensorboard"))
    return model_dir


def get_oldest_model_path(model_dir, model_name, best=True):
    getter = lambda x: int(x.split("_")[-1].split(".")[0])
    model_paths = glob.glob(os.path.join(model_dir, f"{model_name}_*.ckpt"))
    if best:
        path = os.path.join(model_dir, "best_model.ckpt")
        if os.path.exists(path):
            return path
    if model_paths:
        model_paths = sorted(model_paths, key=getter)
        return model_paths[-1]
    else:
        return None


def get_oldest_state(model_dir, name):
    """waiting for optimization, interfaces are not good"""
    best_model_path = get_oldest_model_path(model_dir, name)
    if best_model_path:
        state = torch.load(best_model_path)
        return state, state["epoch"]
    else:
        return None, -1


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding="utf-8")
        return json.JSONEncoder.default(self, obj)


def get_git_hash():
    process = subprocess.Popen(
        ["git", "rev-parse", "HEAD"], shell=False, stdout=subprocess.PIPE
    )
    git_head_hash = process.communicate()[0].strip()
    return git_head_hash


def add_extra_info(config):
    config.git_hash_head = get_git_hash()
    config.host = socket.gethostname()
    config.test_acc = 0
    config.eval_acc = 0
    return dotdict(config.__dict__)


def get_param_path(config):
    expr_name = config.expr_name
    model_dir = check_dir(expr_name)
    param_path = os.path.join(model_dir, "params.json")
    return param_path


def save_info(config):
    param_path = get_param_path(config)
    with open(param_path, "wt", encoding="utf-8") as f:
        json.dump(vars(config), f, indent=4, cls=MyEncoder)


def load_info(config):
    param_path = get_param_path(config)
    with open(param_path, "wt", encoding="utf-8") as f:
        return json.load(f)


def param_save(config, state=None, best_acc=0):
    if state:
        torch.save(state, os.path.join("logs", config.expr_name, "best_model.ckpt"))
    config.best_acc = best_acc
    save_info(config)


def get_bw(config):
    if config.bandwidths:
        if len(config.bandwidths) == 1:
            bw = config.bandwidths[0]
        else:
            bw = config.bandwidths
    else:
        bw = 32
    return bw
