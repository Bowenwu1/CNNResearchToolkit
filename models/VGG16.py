import torch.nn as nn
import torch

__all__ = ["vgg16", "vgg16_bn"]


def vgg16(**kwargs):
    import torchvision

    model = torchvision.models.vgg16(**kwargs)
    return model


def vgg16_bn(**kwargs):
    import torchvision

    model = torchvision.models.vgg16_bn(**kwargs)
    return model


from .wrapper import ModelWrapper


class VGG16_Wrapper(ModelWrapper):
    def __init__(self, opt):
        _net = vgg16(num_classes=opt.num_classes)
        super().__init__(opt, _net)
