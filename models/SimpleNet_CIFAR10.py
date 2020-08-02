"""MobileNet in PyTorch.

See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import random
class Conv2dRandomScale(torch.nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                             padding=0, dilation=1, groups=1, bias=False, use_conv_mask=True):
        super(Conv2dRandomScale, self).__init__(in_channels, out_channels, kernel_size,
                                                                             stride=stride,
                                                                             padding=padding, dilation=dilation,
                                                                             groups=groups, bias=bias)
        self.ori_dilation = dilation
        self.ori_padding = padding

    def train(self, mode=True):
        if not mode:
            self.reset()
        super(Conv2dRandomScale, self).train(mode)

    def switch(self):
        if self.dilation[0] == 1:
            self.dilation = (2, 2)
        else:
            self.dilation = (1, 1)
        self.padding = (self.dilation[0] * (self.kernel_size[0] - 1) // 2, self.dilation[1] * (self.kernel_size[1] - 1) // 2)
    
    def reset(self):
        self.dilation = (self.ori_dilation,) * 2
        self.padding = (self.ori_padding,) * 2

    def forward(self, x):
        return super(Conv2dRandomScale, self).forward(x)

class SwitchBatchNorm2d(torch.nn.Module):
    def __init__(self, num_switches, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(SwitchBatchNorm2d, self).__init__()

        self.bns = nn.ModuleList([torch.nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for i in range(num_switches)])
        self.current_switch = 0
        self.num_switches = num_switches

    def train(self, mode=True):
        if not mode:
            self.reset()
        super(SwitchBatchNorm2d, self).train(mode)

    def switch(self):
        self.current_switch = (self.current_switch + 1) % self.num_switches

    def reset(self):
        self.current_switch = 0

    def forward(self, x):
        return self.bns[self.current_switch](x)

class ConvBNReLU(nn.Module):
    """Depthwise conv + Pointwise conv"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, groups=1, stride=1
    ):
        super(ConvBNReLU, self).__init__()
        self.conv = Conv2dRandomScale(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=groups,
            bias=False,
        )
        self.bn = SwitchBatchNorm2d(2, out_channels) #nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def switch(self):
        self.conv.switch()
        self.bn.switch()
    
    def reset(self):
        self.conv.reset()
        self.bn.reset()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out

class SimpleNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (64, 2), (128, 2), (256, 2)]

    def __init__(self, num_classes=10):
        super(SimpleNet, self).__init__()
        self.layers = self._make_layers(in_channels=3)
        self.linear = nn.Linear(self.cfg[-1][0], num_classes)

    def _make_layers(self, in_channels):
        layers = []
        for x in self.cfg:
            out_channels = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(ConvBNReLU(in_channels, out_channels, stride=stride))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def switch(self):
        def switch_dilation(module):
            if isinstance(module, ConvBNReLU) and random.random() > 0.5:
                module.switch()
        
        self.apply(switch_dilation)
    
    def reset(self):
        def reset_module(module):
            if isinstance(module, ConvBNReLU):
                module.reset()
        
        self.apply(reset_module)   

    def forward(self, x):
        out = self.layers(x)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def test():
    net = SimpleNet()
    x = torch.randn(1, 3, 32, 32)
    print(net)
    y = net(x)
    print(y.size())


# test()
from .wrapper import ModelWrapper


class SimpleNet_Wrapper(ModelWrapper):
    def __init__(self, opt):
        __model = SimpleNet(num_classes=opt.num_classes)
        super().__init__(opt, __model)
