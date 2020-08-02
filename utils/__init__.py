import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F


def seed_torch(seed=None):
    if not isinstance(seed, int):
        return

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def create_optimizer(parameters, opt):
    if opt.type == "SGD":
        return optim.SGD(
            parameters, lr=opt.lr, momentum=0.9, weight_decay=opt.weight_decay
        )
    elif opt.type == "Adam":
        return optim.Adam(
            parameters,
            lr=opt.lr,
            betas=(opt.beta_a, opt.beta_b),
            eps=opt.eps,
            weight_decay=opt.weight_decay,
        )
    else:
        raise Exception("{} optimizer is not supported".format(opt.type))


def create_lr_scheduler(optimizer, opt, epoch):
    if opt.type == "linear":
        return optim.lr_scheduler.MultiStepLR(
            optimizer, opt.milestons, gamma=opt.gamma
        )
    elif opt.type == "cos":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, epoch)
    elif opt.type == "constant":
        # use stepLR with gamma=0 to simulate constant lr
        return optim.lr_scheduler.StepLR(optimizer, 100, 1)
    else:
        raise Exception("{} lr_scheduler is not supported".format(opt.type))


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


import torch
import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        print('#'*20, 'labelsmoothing')
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class LabelSmoothingLoss(nn.Module):
    def __init__(self, eps=0.0):
        self.eps = eps
        super(LabelSmoothingLoss, self).__init__()
    
    def forward(self, x, target):
        n_class = x.size(1)
        one_hot = torch.zeros_like(x).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / (n_class - 1)
        log_prb = F.log_softmax(x, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss

from .config import read_process_config

__all__ = [read_process_config]

############# visualize param distribution #############
import numpy as np
from functools import partial

def write_param(net, writer, epoch, lp=1):
    def _write_single_param(writer, key, param, lp, epoch):
        if param is not None:
            view_channels = param.view(param.size(0), -1)
            mags = lp(view_channels, dim=1)
            writer.add_histogram(key, mags.clone().cpu().data.numpy(), epoch)
            return mags.clone().cpu().data.numpy()
        return np.array([])
    lp_magnitude = partial(torch.norm, p=lp)
    mag = np.array([])
    for key, module in net.named_modules():
        if isinstance(module, nn.modules.conv._ConvNd):
            f_mags = _write_single_param(writer, 'CONV/' + key + '.weight', module.weight, lp_magnitude, epoch)
            mag = np.append(mag, f_mags)
            f_mags = _write_single_param(writer, 'CONV/' + key + '.bias', module.bias, lp_magnitude, epoch)
            mag = np.append(mag, f_mags)
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            _write_single_param(writer, 'BN/' + key + '.weight', module.weight, lp_magnitude, epoch)
            _write_single_param(writer, 'BN/' + key + '.bias', module.bias, lp_magnitude, epoch)
    writer.add_histogram('all-param', mag, epoch)
