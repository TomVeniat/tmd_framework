# Implementation of self-attention layer by Christian Cosgrove
# For SAGAN implementation
# Based on Non-local Neural Networks by Wang et al. https://arxiv.org/abs/1711.07971

import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm

from .non_local import SelfAttention, SelfAttentionMask
from .partial_conv import PartialConv2d
from torch.nn import init

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'dirac':
                init.dirac_(m.weight.data)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 and m.affine:
            print(classname)
            print(m)
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('Init type: {}, gain: {}'.format(init_type, gain))
    net.apply(init_func)


def init_net(net, ngpu=1, **kwargs):
    print('------------ {} ----------'.format(str(net.__class__.__name__)))
    init_type = kwargs.pop('init')
    init_gain = kwargs.pop('init_gain')
    net = net(**kwargs)
    init_weights(net, init_type, gain=init_gain)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters : {:.3f} M'.format(num_params / 1e6))
    print('-----------------------------------------------')
    if ngpu > 1:
        net = torch.nn.parallel.DataParallel(net).cuda()
    else:
        net = net.cuda()
    return net


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return F.avg_pool2d(x, 2)


class ConditionalNorm(nn.Module):
    def __init__(self, num_features, n_condition=148):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)

    def forward(self, input, embed):
        out = self.bn(input)

        if embed is not None:
            gamma, beta = embed.chunk(2, 1)
            gamma = gamma.unsqueeze(2).unsqueeze(3)
            beta = beta.unsqueeze(2).unsqueeze(3)

            if self.num_features < gamma.size(1):
                gamma = gamma[:,:self.num_features]
                beta = beta[:,:self.num_features]
            else:
                gamma = gamma.repeat([1, out.size(1) // gamma.size(1), 1, 1])
                beta = beta.repeat([1, out.size(1) // beta.size(1), 1, 1])

            out = gamma * out + beta

        return out


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes, mask_input):
        super().__init__()

        self.num_classes = num_classes

        self.b1 = ConditionalNorm(num_features=in_channels, n_condition=num_classes)
        self.b2 = ConditionalNorm(num_features=out_channels, n_condition=num_classes)
        self.c1 = spectral_norm(
            PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1, **mask_input))
        self.c2 = spectral_norm(
            PartialConv2d(out_channels, out_channels, kernel_size=3, padding=1, **mask_input))
        self.c_sc = spectral_norm(
            PartialConv2d(in_channels, out_channels, kernel_size=1, padding=0, **mask_input))

    def forward(self, x, embed=None, mask=None):

        if mask is not None:
            mask_h = mask
            h = self.b1(x, embed)
            h = F.relu(h)
            h, mask_h = self.c1(h, mask=mask_h)
            h = self.b2(h, embed)
            h = F.relu(h)
            h, mask_h = self.c2(h, mask_h)
            x, mask = self.c_sc(x, mask)
            return h + x, mask_h

        else:
            h = self.b1(x, embed)
            h = F.relu(h)
            h, = self.c1(h, mask=None)
            h = self.b2(h, embed)
            h = F.relu(h)
            h = self.c2(h, mask=None)
            x = self.c_sc(x, mask)
            return h + x


class ResNetGenerator(nn.Module):
    def __init__(self, ngf, self_attention=True, add_input=False,
                 input_nc=3, output_nc=3, num_class=1000, mask_input=False):
        super().__init__()

        self.mask_input = mask_input
        if self.mask_input:
            mask_input = {'return_mask': True}
        else:
            mask_input = {}

        self.num_class = num_class
        if num_class > 0:

            self.embed = spectral_norm(nn.Embedding(num_class, ngf*4))

        self.self_attention = self_attention
        self.add_input = add_input
        self.block1 = Block(input_nc, ngf * 16, num_classes=num_class, mask_input=mask_input)
        self.block2 = Block(16 * ngf, 8 * ngf, num_classes=num_class, mask_input=mask_input)
        self.block3 = Block(8 * ngf, 4 * ngf, num_classes=num_class, mask_input=mask_input)
        self.block4 = Block(4 * ngf, 2 * ngf, num_classes=num_class, mask_input=mask_input)
        if self_attention:
            self.att = SelfAttentionMask(2 * ngf)
        self.block5 = Block(2 * ngf, ngf, num_classes=num_class, mask_input=mask_input)
        self.b7 = nn.BatchNorm2d(ngf)
        self.l8 = PartialConv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1, **mask_input)

    def forward(self, x, target, mask):
        h = x
        if self.mask_input:
            mask = mask.squeeze()[:, 0:1].float().cuda()
        else:
            mask = None

        if self.num_class > 0:
            emb = self.embed(target)
        else:
            emb = None

        h, mask = self.block1(h, embed=emb, mask=mask)
        h, mask = self.block2(h, embed=emb, mask=mask)
        h, mask = self.block3(h, embed=emb, mask=mask)
        h, mask = self.block4(h, embed=emb, mask=mask)
        if self.self_attention:
            h, mask = self.att(h, mask)
        h, mask = self.block5(h, embed=emb, mask=mask)
        h = self.b7(h)
        h = F.relu(h)
        if self.mask_input:
            h, _ = self.l8(h, mask)
        else:
            h = self.l8(h)
        if self.add_input:
            return torch.tanh(h) + x
        else:
            return torch.tanh(h)


class ResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.c1 = spectral_norm(
            PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.c2 = spectral_norm(
            PartialConv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.c_sc = spectral_norm(
            PartialConv2d(in_channels, out_channels, kernel_size=1, padding=0))

    def forward(self, x):
        h = F.relu(x)
        h = self.c1(h)
        h = F.relu(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        x = self.c_sc(x)
        if self.downsample:
            x = _downsample(x)
        return h + x


class OptiResBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.c1 = spectral_norm(
            PartialConv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.c2 = spectral_norm(
            PartialConv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.c_sc = spectral_norm(
            PartialConv2d(in_channels, out_channels, kernel_size=1, padding=0))

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.relu(h)
        h = self.c2(h)
        h = _downsample(h)
        x = self.c_sc(_downsample(x))
        return h + x


class ResNetDiscriminator(nn.Module):
    def __init__(self, ndf, self_attention=True, input_nc=3, num_class=1000):
        super().__init__()
        self.self_attention = self_attention
        self.num_class = num_class

        self.block0 = OptiResBlockDown(input_nc, ndf)
        if self_attention:
            self.att = SelfAttention(ndf)
        self.block1 = ResBlockDown(
            ndf, 2 * ndf, downsample=True)
        self.block2 = ResBlockDown(
            2 * ndf, 4 * ndf, downsample=True)
        self.block3 = ResBlockDown(
            4 * ndf, 8 * ndf, downsample=True)
        self.block4 = ResBlockDown(
            8 * ndf, 16 * ndf, downsample=True)
        self.block5 = ResBlockDown(
            16 * ndf, 16 * ndf, downsample=True)

        self.l6 = spectral_norm(nn.Linear(ndf * 16, 1))

        if num_class > 0:
            self.embed = spectral_norm(nn.Embedding(num_class, 16 * ndf))

    def forward(self, x, target):
        h = self.block0(x)
        if self.self_attention:
            h = self.att(h)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = F.relu(h)
        if target is not None and self.num_class > 0:
            embed = self.embed(target)
            h = torch.sum(h, dim=(2, 3))  # Global pooling
            h = embed * h
        else:
            h = torch.sum(h, dim=(2, 3))  # Global pooling
        h = torch.sigmoid(self.l6(h))
        return h
