import torch
from torch import nn
import torch.nn.functional as F

from .network import Network
from nets import net_utils
from model.config import cfg


class mCReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, same_padding=False):
        super(mCReLU, self).__init__()
        out1, out2, out3 = out_channels
        self.residual = out1 is not None and out3 is not None
        # momentum = 0.05 if self.training else 0

        if self.residual:
            self.conv0 = net_utils.Conv2d(in_channels, out3, 1, stride, relu=False,
                                          same_padding=same_padding) if in_channels != out3 else None
            self.conv1 = net_utils.Conv2d_BatchNorm(in_channels, out1, 1, same_padding=same_padding)
            self.conv2 = net_utils.Conv2d_CReLU(out1, out2, kernel_size, stride, same_padding=same_padding)
            self.conv3 = net_utils.Conv2d(out2*2, out3, 1, relu=False, same_padding=same_padding)
            self.bn = nn.BatchNorm2d(in_channels)
        else:
            self.conv2 = net_utils.Conv2d_CReLU(
                in_channels, out2, kernel_size, stride, same_padding=same_padding, bias=False)

    def forward(self, x):
        if not self.residual:
            return self.conv2(x)

        x_bn = F.relu(self.bn(x))

        x0 = x if self.conv0 is None else self.conv0(x_bn)

        x1 = self.conv1(x_bn)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x_post = x0 + x3

        return x_post


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Inception, self).__init__()
        # momentum = 0.05 if self.training else 0
        out_1x1, out_3x3, out_5x5, out_pool, out_out = out_channels

        self.conv1 = nn.Sequential(
            net_utils.Conv2d_BatchNorm(in_channels, out_1x1, 1, stride, same_padding=True)
        )

        self.conv2 = nn.Sequential(
            net_utils.Conv2d_BatchNorm(in_channels, out_3x3[0], 1, stride, same_padding=True),
            net_utils.Conv2d_BatchNorm(out_3x3[0], out_3x3[1], 3, 1, same_padding=True)
        )

        self.conv3 = nn.Sequential(
            net_utils.Conv2d_BatchNorm(in_channels, out_5x5[0], 1, stride, same_padding=True),
            net_utils.Conv2d_BatchNorm(out_5x5[0], out_5x5[1], 3, 1, same_padding=True),
            net_utils.Conv2d_BatchNorm(out_5x5[1], out_5x5[2], 3, 1, same_padding=True)
        )

        self.conv_pool = nn.Sequential(
            nn.MaxPool2d(3, stride, padding=1),
            net_utils.Conv2d_BatchNorm(in_channels, out_pool, 1, 1, same_padding=True)
        ) if out_pool > 0 else None

        in_c = out_1x1 + out_3x3[-1] + out_5x5[-1] + out_pool
        self.conv_out = net_utils.Conv2d(in_c, out_out, 1, relu=False, same_padding=True)

        self.conv0 = net_utils.Conv2d(in_channels, out_out, 1, stride, relu=False,
                                      same_padding=True) if in_channels != out_out else None
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x_bn = F.relu(self.bn(x))

        x1 = self.conv1(x_bn)
        x2 = self.conv2(x_bn)
        x3 = self.conv3(x_bn)
        xs = [x1, x2, x3]
        if self.conv_pool is not None:
            xs.append(self.conv_pool(x_bn))

        x4 = self.conv_out(torch.cat(xs, 1))
        x0 = x if self.conv0 is None else self.conv0(x_bn)
        x_post = x0 + x4

        return x_post


def _make_layers(in_channels, net_cfg):
    layers = []

    if len(net_cfg) > 0 and isinstance(net_cfg[0], list):
        for sub_cfg in net_cfg:
            layer, in_channels = _make_layers(in_channels, sub_cfg)
            layers.append(layer)
    else:
        for item in net_cfg:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif item[0] == 'C':
                out_channels, ksize, stride = item[1:]
                layers.append(mCReLU(in_channels, out_channels, ksize, stride, same_padding=True))
                in_channels = out_channels[2] if out_channels[2] is not None else out_channels[1] * 2
            elif item[0] == 'I':
                out_channels, stride = item[1:]
                layers.append(Inception(in_channels, out_channels, stride))
                in_channels = out_channels[-1]
            else:
                assert False, 'Unknown layer type'

    return nn.Sequential(*layers), in_channels


class PVANet(nn.Module):
    def __init__(self):
        super(PVANet, self).__init__()

        net_cfgs = [
            # conv1s
            [('C', (None, 16, None), 7, 2), 'M'],
            [('C', (24, 24, 64), 3, 1), ('C', (24, 24, 64), 3, 1), ('C', (24, 24, 64), 3, 1)],
            [('C', (48, 48, 128), 3, 2), ('C', (48, 48, 128), 3, 1), ('C', (48, 48, 128), 3, 1), ('C', (48, 48, 128), 3, 1)],

            [('I', (64, (48, 128), (24, 48, 48), 128, 256), 2),
             ('I', (64, (64, 128), (24, 48, 48), 0, 256), 1),
             ('I', (64, (64, 128), (24, 48, 48), 0, 256), 1),
             ('I', (64, (64, 128), (24, 48, 48), 0, 256), 1)],

            [('I', (64, (96, 192), (32, 64, 64), 128, 384), 2),
             ('I', (64, (96, 192), (32, 64, 64), 0, 384), 1),
             ('I', (64, (96, 192), (32, 64, 64), 0, 384), 1),
             ('I', (64, (96, 192), (32, 64, 64), 0, 384), 1)],
        ]

        in_channels = 3
        self.conv1, in_channels = _make_layers(in_channels, net_cfgs[0])
        self.conv2, in_channels = _make_layers(in_channels, net_cfgs[1])
        self.conv3, conv3_c = _make_layers(in_channels, net_cfgs[2])

        self.conv4, conv4_c = _make_layers(conv3_c, net_cfgs[3])
        self.conv5, conv5_c = _make_layers(conv4_c, net_cfgs[4])

        self.downscale = nn.MaxPool2d(3, 2, padding=1)
        self.upscale = nn.ConvTranspose2d(conv5_c, conv5_c, 4, 2, padding=1, groups=in_channels, bias=False)
        self.convf = net_utils.Conv2d(conv3_c + conv4_c + conv5_c, 512, 1, same_padding=True)

        self.feat_stride = 16
        self.out_channels = 512

    def forward(self, image):
        conv1 = self.conv1(image)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        dscale = self.downscale(conv3)
        uscale = self.upscale(conv5)
        # print conv5.size()
        # print uscale.size(), conv4.size(), dscale.size()
        convf = self.convf(torch.cat([uscale, conv4, dscale], 1))

        return convf


class pvanet(Network):
    def __init__(self):
        Network.__init__(self)
        self._feat_stride = [16, ]
        self._feat_compress = [1. / float(self._feat_stride[0]), ]
        self._net_conv_channels = 512
        self._fc7_channels = 1024

    def _crop_pool_layer(self, bottom, rois):
        return Network._crop_pool_layer(self, bottom, rois, cfg.RESNET.MAX_POOL)

    def _image_to_head(self):
        net_conv = self._layers['head'](self._image)
        self._act_summaries['conv'] = net_conv

        return net_conv

    def _head_to_tail(self, pool5):
        pool5_flat = F.avg_pool2d(pool5, pool5.size()[2:4]).view(pool5.size(0), -1)
        # print(pool5_flat.size())
        fc7 = self.fc7(pool5_flat)

        return fc7

    def _init_head_tail(self):
        self.pvanet = PVANet()
        self.fc7 = nn.Sequential(
            nn.Linear(self.pvanet.out_channels, self._fc7_channels),
            nn.ReLU(),
        )

        # Fix blocks
        fixed_blocks = cfg.PVANET.FIXED_BLOCKS
        assert (0 <= fixed_blocks < 4)
        if fixed_blocks >= 3:
            for p in self.pvanet.conv3.parameters(): p.requires_grad = False
        if fixed_blocks >= 2:
            for p in self.pvanet.conv2.parameters(): p.requires_grad = False
        if fixed_blocks >= 1:
            for p in self.pvanet.conv1.parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.pvanet.apply(set_bn_fix)

        # Build pvanet.
        self._layers['head'] = self.pvanet

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode (not really doing anything)
            self.pvanet.eval()
            if cfg.RESNET.FIXED_BLOCKS <= 3:
                self.pvanet.conv4.train()
            if cfg.RESNET.FIXED_BLOCKS <= 2:
                self.pvanet.conv3.train()
            if cfg.RESNET.FIXED_BLOCKS <= 1:
                self.pvanet.conv2.train()
            if cfg.RESNET.FIXED_BLOCKS == 0:
                self.pvanet.conv1.train()

            # Set batchnorm always in eval mode during training
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.pvanet.apply(set_bn_eval)

    def load_pretrained_cnn(self, h5file):
        net_utils.load_net(h5file, self.pvanet)
