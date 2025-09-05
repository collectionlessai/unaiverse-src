"""
       █████  █████ ██████   █████           █████ █████   █████ ██████████ ███████████    █████████  ██████████
      ░░███  ░░███ ░░██████ ░░███           ░░███ ░░███   ░░███ ░░███░░░░░█░░███░░░░░███  ███░░░░░███░░███░░░░░█
       ░███   ░███  ░███░███ ░███   ██████   ░███  ░███    ░███  ░███  █ ░  ░███    ░███ ░███    ░░░  ░███  █ ░ 
       ░███   ░███  ░███░░███░███  ░░░░░███  ░███  ░███    ░███  ░██████    ░██████████  ░░█████████  ░██████   
       ░███   ░███  ░███ ░░██████   ███████  ░███  ░░███   ███   ░███░░█    ░███░░░░░███  ░░░░░░░░███ ░███░░█   
       ░███   ░███  ░███  ░░█████  ███░░███  ░███   ░░░█████░    ░███ ░   █ ░███    ░███  ███    ░███ ░███ ░   █
       ░░████████   █████  ░░█████░░████████ █████    ░░███      ██████████ █████   █████░░█████████  ██████████
        ░░░░░░░░   ░░░░░    ░░░░░  ░░░░░░░░ ░░░░░      ░░░      ░░░░░░░░░░ ░░░░░   ░░░░░  ░░░░░░░░░  ░░░░░░░░░░ 
                 A Collectionless AI Project (https://collectionless.ai)
                 Registration/Login: https://unaiverse.io
                 Code Repositories:  https://github.com/collectionlessai/
                 Main Developers:    Stefano Melacci (Project Leader), Christian Di Maio, Tommaso Guidi
"""
import math
import torch
from .cnus import CNUs
import torch.nn.functional as F
from collections.abc import Iterable


class LinearCNU(CNUs):
    def __init__(self, in_features, out_features, bias=True, device=None,
                 shared_keys=True, key_mem_units=2, psi_fn='identity', key_size=None, **kwargs):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.shared_keys = shared_keys

        if kwargs is not None:
            assert 'q' not in kwargs, "The number of CNUs is automatically determined, do not set argument 'q'"
            assert 'd' not in kwargs, "The size of each key can be specified with argument 'key_size', " \
                                      "do not set argument 'd'"
            assert 'm' not in kwargs, "The number of keys and memory units can be specified with argument " \
                                      "'key_mem_units', do not set argument 'm'"
            assert 'u' not in kwargs, "Size of each memory unit is automatically determined, do not set argument 'u'"

        # Number of keys/memory units
        kwargs['m'] = key_mem_units

        # Size of each key
        kwargs['d'] = in_features if key_size is None else key_size

        # Function used to compare input against keys
        kwargs['psi_fn'] = psi_fn

        if not shared_keys:

            # Each neuron is an independent cnu, with its own keys and its own memory units
            kwargs['q'] = self.out_features
            kwargs['u'] = self.in_features + (1 if self.bias else 0)
        else:

            # All the CNUs of the layer share the same keys, thus their memory units are concatenated
            kwargs['q'] = 1
            kwargs['u'] = self.out_features * (self.in_features + (1 if self.bias else 0))

        # Creating neurons
        super(LinearCNU, self).__init__(**kwargs)

        # Switching device
        if device is not None:
            self.to(device)

        # Clearing
        if not self.bias:
            self.bias = None

    def forward(self, x):

        # Getting weights
        W = self.compute_weights(x)

        # Ensuring the shape is right (needed when neurons share the same keys)
        W = W.reshape((x.shape[0], self.out_features, -1))  # [b,q,1] => [b, out_features,(in_features + 1-if-bias)]

        # Splitting into weights and biases
        if self.bias:
            weights = W[:, :, :-1]  # [b,out_features,in_features]
            bias = W[:, :, -1]  # [b,out_features]
        else:
            weights = W  # [b,out_features,in_features]
            bias = None

        # Batched linear projection: matmul([b,out_features,in_features], [b,in_features,1]) = [b,out_features,1]
        # that we squeeze to [b,out_features]
        o = torch.matmul(weights, x.unsqueeze(2)).squeeze(2)  # [b,out_features]
        if bias is not None:
            o += bias
        return o

    def reset_parameters(self):
        self.reset_memories = False
        super().reset_parameters()

        # We ensure that memories M are initialized as Pytorch does for the classic linear layer
        q = self.M.shape[0]
        m = self.M.shape[1]
        self.M.data.zero_()  # Ensures we don’t keep old values

        for j in range(q):
            for i in range(m):

                # Initialize weight and bias separately for each memory
                weight = torch.empty(self.out_features if self.shared_keys else 1, self.in_features)
                torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))  # Computes fan in

                if self.bias:
                    bias = torch.empty(self.out_features if self.shared_keys else 1)
                    bound = 1 / math.sqrt(self.in_features)
                    torch.nn.init.uniform_(bias, -bound, bound)
                    weight_bias = torch.cat([weight, bias.unsqueeze(1)], dim=1)
                else:
                    weight_bias = weight

                # Store the flattened weight_bias into self.M[i]
                self.M.data[j, i, :] = weight_bias.flatten()

    def __str__(self):
        s = "- in_features = " + str(self.in_features) + "\n"
        s += "- out_features = " + str(self.out_features) + "\n"
        s += "- bias = " + str(self.bias) + "\n"
        return "[cnu-based Linear Layer]\n" + s + super().__str__()


class Conv2d(CNUs):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, padding_mode='zeros',
                 dilation=1, groups=1, bias=True, device=None,
                 shared_keys=True, key_mem_units=2, psi_fn='reduce2d', key_size=None, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, Iterable) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, Iterable) else (stride, stride)
        self.padding = padding
        self.padding_mode = padding_mode
        self.dilation = dilation if isinstance(dilation, Iterable) else (dilation, dilation)
        self.groups = groups
        self.bias = bias
        self.in_features = math.prod(self.kernel_size) * self.in_channels

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(valid_padding_modes,
                                                                                                padding_mode))
        if isinstance(padding, str):
            self.__reversed_padding_repeated_twice = [0, 0] * len(self.kernel_size)
            if padding == 'same':
                for d, k, i in zip(self.dilation, self.kernel_size,
                                   range(len(self.kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self.__reversed_padding_repeated_twice[2 * i] = left_pad
                    self.__reversed_padding_repeated_twice[2 * i + 1] = (total_padding - left_pad)
        else:
            self.padding = padding if isinstance(padding, Iterable) else (padding, padding)
            self.__reversed_padding_repeated_twice = tuple(x for x in reversed(self.padding) for _ in range(2))

        if kwargs is not None:
            assert 'q' not in kwargs, "The number of CNUs is automatically determined, do not set argument 'q'"
            assert 'd' not in kwargs, "The size of each key can be specified with argument 'key_size', " \
                                      "do not set argument 'd'"
            assert 'm' not in kwargs, "The number of keys and memory units can be specified with argument " \
                                      "'key_mem_units', do not set argument 'm'"
            assert 'u' not in kwargs, "Size of each memory unit is automatically determined, do not set argument 'u'"

        # Number of keys/memory units
        kwargs['m'] = key_mem_units

        # Size of each key
        if key_size is not None:
            if isinstance(key_size, (tuple, list)):
                key_size = math.prod(key_size)
            kwargs['d'] = key_size
        else:
            kwargs['d'] = (5 * 5 * self.in_channels)

        # Function used to compare input against keys
        kwargs['psi_fn'] = psi_fn

        if not shared_keys:

            # Each neuron is an independent cnu, with its own keys and its own memory units
            kwargs['q'] = self.out_channels
            kwargs['u'] = self.in_features + (1 if self.bias else 0)
        else:

            # All the CNUs of the layer share the same keys, thus their memory units are concatenated
            kwargs['q'] = 1
            kwargs['u'] = self.out_channels * (self.in_features + (1 if self.bias else 0))

        # Creating neurons
        super(Conv2d, self).__init__(**kwargs)

        # Switching device
        if device is not None:
            self.to(device)

    def forward(self, x):

        # Shortcuts
        b, c, h, w = x.shape

        # Getting weights
        W = self.compute_weights(x)

        # Ensuring the shape is right (needed when neurons share the same keys)
        W = W.reshape((b, self.out_channels, -1))  # [b,q,1] => [b,out_channels,(in_features + 1-if-bias)]

        # Splitting into weights and biases
        if self.bias:
            weights = W[:, :, :-1]  # [b,out_channels,in_features]
            bias = W[:, :, -1]  # [b,out_channels]
        else:
            weights = W  # [b,out_channels,in_features]
            bias = None

        # Creating tensor with convolutional filters
        kernels = self.__mat2filters(weights)

        # Stack all images along the channels
        x = x.view(1, b * c, h, w)

        # Convolution
        if self.padding_mode != 'zeros':
            x = F.conv2d(F.pad(x, self.__reversed_padding_repeated_twice, mode=self.padding_mode),
                         kernels, bias.flatten() if bias is not None else None, self.stride,
                         (0, 0), self.dilation, groups=(b * self.groups))
        else:
            x = F.conv2d(x, kernels, bias.flatten() if bias is not None else None, self.stride,
                         self.padding, self.dilation, groups=(b * self.groups))

        return x.view(b, self.out_channels, x.shape[2], x.shape[3])

    def __mat2filters(self, weights):
        """
        :param weights: tensor with blended memories (weights) with shape [b,out_channels,in_features]
        """
        if type(self.kernel_size) is tuple:
            kernel_size_h, kernel_size_w = self.kernel_size
        else:
            kernel_size_h = self.kernel_size
            kernel_size_w = self.kernel_size
        b = weights.shape[0]
        out_channels = b * weights.shape[1]
        receptive_field_volume = weights.shape[2]
        in_channels_div_b_times_groups = receptive_field_volume // (kernel_size_h * kernel_size_w)
        return weights.reshape(out_channels, in_channels_div_b_times_groups, kernel_size_h, kernel_size_w)

    def __str__(self):
        s = "- in_channels = " + str(self.in_channels) + "\n"
        s += "- out_channels = " + str(self.out_channels) + "\n"
        s += "- kernel_size = " + str(self.kernel_size) + "\n"
        s += "- stride = " + str(self.stride) + "\n"
        s += "- padding = " + str(self.padding) + "\n"
        s += "- padding_mode = " + str(self.padding_mode) + "\n"
        s += "- dilation = " + str(self.dilation) + "\n"
        s += "- groups = " + str(self.groups) + "\n"
        s += "- bias = " + str(self.bias) + "\n"
        s += "- in_features = " + str(self.in_features) + "\n"
        return "[cnu-based Conv2d Layer]\n" + s + super().__str__()
