import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
from torch.legacy.nn.SpatialUpSamplingNearest import SpatialUpSamplingNearest

import numpy as np
from copy import deepcopy
import collections
try:
    import cPickle as pickle
except ImportError:
    import pickle


class UpSamplingCaffe(nn.Module):
    def __init__(self, in_channels, scale_factor=2, require_grad=False):
        super(UpSamplingCaffe, self).__init__()

        self.in_channels = in_channels
        self.kernel_size = int(2 * scale_factor - scale_factor % 2)
        self.stride = scale_factor
        self.pad = int(np.ceil((scale_factor - 1) / 2.))

        self.convt = nn.ConvTranspose2d(in_channels, in_channels, self.kernel_size, self.stride, self.pad,
                                        groups=in_channels, bias=False)
        self.convt.weight.requires_grad = require_grad
        self.reset_parameters()

    def reset_parameters(self):
        self.convt.weight.data.copy_(self.make_bilinear_weights(self.kernel_size, self.in_channels))

    def forward(self, x):
        x = self.convt(x)
        return x

    @staticmethod
    def make_bilinear_weights(size, num_channels):
        """
         Make a 2D bilinear kernel suitable for upsampling
        Stack the bilinear kernel for application to tensor
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filt = (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)
        filt = torch.from_numpy(filt)
        w = torch.zeros(num_channels, 1, size, size)
        for i in range(num_channels):
            w[i, 0] = filt
        return w


class IdentityLayer(nn.Module):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x


class ConcatAddTable(nn.Module):
    def __init__(self, *args):
        super(ConcatAddTable, self).__init__()
        if len(args) == 1 and isinstance(args[0], collections.OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            idx = 0
            for module in args:
                self.add_module(str(idx), module)
                idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def forward(self, input):
        x_out = None
        for module in self._modules.values():
            x = module(input)
            if x_out is None:
                x_out = x
            else:
                x_out = x_out + x
        return x_out


class ConcatTable(nn.Module):
    def __init__(self, *args):
        super(ConcatTable, self).__init__()
        if len(args) == 1 and isinstance(args[0], collections.OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            idx = 0
            for module in args:
                self.add_module(str(idx), module)
                idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def forward(self, input):
        xs = []
        for module in self._modules.values():
            xs.append(module(input))
        return torch.cat(xs, 1)


class SeqConcatTable(nn.Module):
    def __init__(self, *args):
        super(SeqConcatTable, self).__init__()
        if len(args) == 1 and isinstance(args[0], collections.OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            idx = 0
            for module in args:
                self.add_module(str(idx), module)
                idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def forward(self, input):
        xs = []
        for module in self._modules.values():
            xs.append(module(input))
            input = xs[-1]
        return torch.cat(xs, 1)


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(SeparableConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.pointwise(x)
        x = F.relu(self.bn2(x), inplace=True)
        return x


class Conv2d_BatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bias=False):
        super(Conv2d_BatchNorm, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        # momentum = 0.05 if self.training else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Conv2d_CReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, same_padding=False, bias=False):
        super(Conv2d_CReLU, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        # momentum = 0.05 if self.training else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.scale = Scale(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = torch.cat([x, -x], 1)
        x = self.scale(x)
        return self.relu(x)


class Scale(nn.Module):
    def __init__(self, in_channels):
        super(Scale, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, x):
        return x * self.alpha.expand_as(x) + self.beta.expand_as(x)


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class UpSamplingFunction(Function):
    def __init__(self, scale):
        self.scale = scale
        self.up = SpatialUpSamplingNearest(scale)

    def forward(self, x):
        self.save_for_backward(x)

        if x.is_cuda:
            self.up.cuda()
        return self.up.updateOutput(x)

    def backward(self, grad_output):
        return self.up.updateGradInput(self.saved_tensors[0], grad_output)


class UpSampling(nn.Module):
    def __init__(self, scale):
        super(UpSampling, self).__init__()
        self.scale = scale

    def forward(self, x):
        return UpSamplingFunction(self.scale)(x)


def str_is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def set_optimizer_state_devices(state, device_id=None):
    """
    set state in optimizer to a device. move to cpu if device_id==None
    :param state: optimizer.state
    :param device_id: None or a number
    :return:
    """
    for k, v in state.items():
        for k2 in v.keys():
            if hasattr(v[k2], 'cuda'):
                if device_id is None:
                    v[k2] = v[k2].cpu()
                else:
                    v[k2] = v[k2].cuda(device_id)

    return state


def save_net(fname, net, epoch=-1, optimizers=None, rm_prev_opt=False):
    import h5py
    with h5py.File(fname, mode='w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
        h5f.attrs['epoch'] = epoch

    if optimizers is not None:
        state_dicts = []
        for optimizer in optimizers:
            state_dict = deepcopy(optimizer.state_dict())
            state_dict['state'] = set_optimizer_state_devices(state_dict['state'], device_id=None)
            state_dicts.append(state_dict)

        state_file = fname + '.optimizer_state.pk'
        with open(state_file, 'w') as f:
            pickle.dump(state_dicts, f)

        # remove
        if rm_prev_opt:
            root = os.path.split(fname)[0]
            for filename in os.listdir(root):
                filename = os.path.join(root, filename)
                if filename.endswith('.optimizer_state.pk') and filename != state_file:
                    print('remove {}'.format(filename))
                    os.remove(filename)


def load_net(fname, net, prefix='', load_state_dict=False):
    import h5py
    with h5py.File(fname, mode='r') as h5f:
        h5f_is_module = True
        for k in h5f.keys():
            if not str(k).startswith('module.'):
                h5f_is_module = False
                break
        if prefix == '' and not isinstance(net, nn.DataParallel) and h5f_is_module:
            prefix = 'module.'

        for k, v in net.state_dict().items():
            k = prefix + k
            if k in h5f:
                param = torch.from_numpy(np.asarray(h5f[k]))
                if v.size() != param.size():
                    print('inconsistent shape: {}, {}'.format(v.size(), param.size()))
                else:
                    v.copy_(param)
            else:
                print('no layer: {}'.format(k))

        epoch = h5f.attrs['epoch'] if 'epoch' in h5f.attrs else -1

        if not load_state_dict:
            if 'learning_rates' in h5f.attrs:
                lr = h5f.attrs['learning_rates']
            else:
                lr = h5f.attrs.get('lr', -1)
                lr = np.asarray([lr] if lr > 0 else [], dtype=np.float)

            return epoch, lr

        state_file = fname + '.optimizer_state.pk'
        if os.path.isfile(state_file):
            with open(state_file, 'r') as f:
                state_dicts = pickle.load(f)
                if not isinstance(state_dicts, list):
                    state_dicts = [state_dicts]
        else:
            state_dicts = None
        return epoch, state_dicts


def is_cuda(model):
    p = next(model.parameters())
    return p.is_cuda


def get_device(model):
    if is_cuda(model):
        p = next(model.parameters())
        return p.get_device()
    else:
        return None


def plot_graph(top_var, fname, params=None):
    """
    This method don't support release v0.1.12 caused by a bug fixed in: https://github.com/pytorch/pytorch/pull/1016
    So if you want to use `plot_graph`, you have to build from master branch or wait for next release.

    Plot the graph. Make sure that require_grad=True and volatile=False
    :param top_var: network output Varibale
    :param fname: file name
    :param params: dict of (name, Variable) to add names to node that
    :return: png filename
    """
    from graphviz import Digraph
    import pydot
    dot = Digraph(comment='LRP',
                  node_attr={'style': 'filled', 'shape': 'box'})
    # , 'fillcolor': 'lightblue'})

    seen = set()

    if params is not None:
        assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = '{}\n '.format(param_map[id(u)]) if params is not None else ''
                node_name = '{}{}'.format(name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    add_nodes(top_var.grad_fn)
    dot.save(fname)
    (graph,) = pydot.graph_from_dot_file(fname)
    im_name = '{}.png'.format(fname)
    graph.write_png(im_name)
    print(im_name)

    return im_name


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor, volatile=False, device_id=None):
    v = Variable(torch.from_numpy(x).type(dtype), volatile=volatile)
    if is_cuda:
        v = v.cuda(device_id)
    return v


def variable_to_np_tf(x):
    return x.data.cpu().numpy().transpose([0, 2, 3, 1])


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            p.grad.mul_(norm)


class CrossEntropy(nn.Module):

    def __init__(self, weight=None, size_average=False, ignore_index=-100):
        """
        CrossEntropy without sum
        :param weight: [C] tensor
        :param size_average:
        :param ignore_index: an int value
        """
        super(CrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        self.weight = weight
        if self.weight is not None:
            self.weight = self.weight.view(-1, 1)
            if self.ignore_index >= 0:
                self.weight[self.ignore_index] = 0
        self.size_average = size_average

    def forward(self, pred, target):
        """

        :param pred: NxCxHxW
        :param target: NxHxW LongTensor
        :return: NxHxW or 1
        """
        bsize, c, h, w = pred.size()
        pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, c)

        logp = F.log_softmax(pred_flat)
        y = target.data.view(-1, 1)
        ymask = logp.data.new(logp.size()).zero_()  # (NxHxW,C) all zero

        if self.weight is None:
            self.weight = torch.ones(c, 1)
            if self.ignore_index >= 0:
                self.weight[self.ignore_index] = 0

        if pred_flat.is_cuda:
            weights = self.weight.cuda(pred_flat.get_device())
        else:
            weights = self.weight
        weights = torch.gather(weights, dim=0, index=y)

        ymask.scatter_(1, y, weights)  # have to make y into shape (NxHxW,1) for scatter_ to be happy
        ymask = Variable(ymask)

        # pluck
        logpy_flat = -(logp * ymask).sum(1)   # [NxHxW]
        logpy = logpy_flat.view(bsize, h, w)

        return logpy


def binary_cross_entropy_with_logits(input, target, weight=None):
    r"""Function that measures Binary Cross Entropy between target and output
    logits:

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Variable of arbitrary shape
        target: Variable of the same shape as input
        weight (Variable, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
    """
    if not target.is_same_size(input):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    return loss
