import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import Adam
import numpy
from einops import rearrange
import time
from transformer import Transformer
from Intra_MLP import index_points, knn_l2

# vgg choice
base = {'vgg': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}


# vgg16
def vgg(cfg, i=3, batch_norm=True):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


def hsp(in_channel, out_channel):
    layers = nn.Sequential(nn.Conv2d(in_channel, out_channel, 1, 1),
                           nn.ReLU())
    return layers


def cls_modulation_branch(in_channel, hiden_channel):
    layers = nn.Sequential(nn.Linear(in_channel, hiden_channel),
                           nn.ReLU())
    return layers


def cls_branch(hiden_channel, class_num):
    layers = nn.Sequential(nn.Linear(hiden_channel, class_num),
                           nn.Sigmoid())
    return layers


def intra():
    layers = []
    layers += [nn.Conv2d(512, 512, 1, 1)]
    layers += [nn.Sigmoid()]
    return layers


def concat_r():
    layers = []
    layers += [nn.Conv2d(512, 512, 1, 1)]
    layers += [nn.ReLU()]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.ReLU()]
    layers += [nn.ConvTranspose2d(512, 512, 4, 2, 1)]
    return layers


def concat_1():
    layers = []
    layers += [nn.Conv2d(512, 512, 1, 1)]
    layers += [nn.ReLU()]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.ReLU()]
    return layers


def mask_branch():
    layers = []
    layers += [nn.Conv2d(512, 2, 3, 1, 1)]
    layers += [nn.ConvTranspose2d(2, 2, 8, 4, 2)]
    layers += [nn.Softmax2d()]
    return layers


def incr_channel():
    layers = []
    layers += [nn.Conv2d(128, 512, 3, 1, 1)]
    layers += [nn.Conv2d(256, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    return layers


def incr_channel2():
    layers = []
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.Conv2d(512, 512, 3, 1, 1)]
    layers += [nn.ReLU()]
    return layers


def norm(x, dim):
    squared_norm = (x ** 2).sum(dim=dim, keepdim=True)
    normed = x / torch.sqrt(squared_norm)
    return normed


def fuse_hsp(x, p, group_size=5):
    t = torch.zeros(group_size, x.size(1))
    for i in range(x.size(0)):
        tmp = x[i, :]
        if i == 0:
            nx = tmp.expand_as(t)
        else:
            nx = torch.cat(([nx, tmp.expand_as(t)]), dim=0)
    nx = nx.view(x.size(0) * group_size, x.size(1), 1, 1)
    y = nx.expand_as(p)
    return y