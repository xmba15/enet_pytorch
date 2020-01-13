#!/usr/bin/env python
# -*- coding: utf-8 -*-
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BlockBase(nn.Module):
    def __init__(
        self,
        input_channels=None,
        output_channels=None,
        regularlizer_prob=0.1,
        internal_ratio=4,
        downsampling=False,
        upsampling=False,
        dilated=False,
        dilation_rate=None,
        asymmetric=False,
        use_relu=False,
        use_bias=False,
        **kwargs
    ):
        super(BlockBase, self).__init__()

        self._input_channels = input_channels
        self._output_channels = output_channels
        self._regularlizer_prob = regularlizer_prob
        self._internal_ratio = internal_ratio
        self._downsampling = downsampling
        self._upsampling = upsampling
        self._dilated = dilated
        self._dilation_rate = dilation_rate
        self._asymmetric = asymmetric
        self._use_relu = use_relu
        self._use_bias = use_bias

        for key, value in kwargs.items():
            assert not key.startswith("_")
            self.__dict__["_{}".format(key)] = value

        if self._input_channels is not None and self._output_channels is not None:
            self._reduced_channels = self._input_channels // 4

        self._input_stride = 2 if self._downsampling else 1

    @property
    def downsampling(self):
        return self._downsampling

    def _activation_func(self, channels, use_relu):
        return nn.PReLU(channels) if not use_relu else nn.ReLU()

    def _block_activation_func(self, channels):
        assert "_use_relu" in self.__dict__

        return self._activation_func(channels, self._use_relu)


class InitialBlock(BlockBase):
    def __init__(self, use_relu=False, use_bias=False):
        super(InitialBlock, self).__init__(use_relu=use_relu, use_bias=use_bias)
        self._conv = nn.Conv2d(3, 13, (3, 3), stride=2, padding=1)
        self._batch_norm = nn.BatchNorm2d(13, 1e-3)
        self._activation = self._block_activation_func(channels=13)
        self._max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        output = torch.cat([self._activation(self._batch_norm(self._conv(x))), self._max_pool(x)], 1)

        return output


class BottleNeck(BlockBase):
    def __init__(
        self,
        input_channels=None,
        output_channels=None,
        regularlizer_prob=0.1,
        internal_ratio=4,
        downsampling=False,
        upsampling=False,
        dilated=False,
        dilation_rate=None,
        asymmetric=False,
        use_relu=False,
        use_bias=False,
    ):
        super(BottleNeck, self).__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            regularlizer_prob=regularlizer_prob,
            internal_ratio=internal_ratio,
            downsampling=downsampling,
            upsampling=upsampling,
            dilated=dilated,
            dilation_rate=dilation_rate,
            asymmetric=asymmetric,
            use_relu=use_relu,
            use_bias=use_bias,
        )

        # fmt: off
        self._block1x1_1 = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(self._input_channels, self._reduced_channels,
                            self._input_stride, self._input_stride, bias=self._use_bias)),
            ("batch_norm", nn.BatchNorm2d(self._reduced_channels, 1e-3)),
            ("block_activation", self._block_activation_func(self._reduced_channels))
        ]))
        # fmt: on

        # convolution for middle block
        conv = nn.Conv2d(self._reduced_channels, self._reduced_channels, 3, stride=1, padding=1)
        if self._downsampling:
            self._pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        elif self._upsampling:
            spatial_conv = nn.Conv2d(self._input_channels, self._output_channels, 1, bias=False)
            batch_norm = nn.BatchNorm2d(self._output_channels, 1e-3)
            self._conv_before_unpool = nn.Sequential(spatial_conv, batch_norm)
            self._unpool = nn.MaxUnpool2d(2)
            conv = nn.ConvTranspose2d(
                self._reduced_channels, self._reduced_channels, 3, stride=2, padding=1, output_padding=1,
            )
        elif self._dilated:
            conv = nn.Conv2d(
                self._reduced_channels,
                self._reduced_channels,
                3,
                padding=self._dilation_rate,
                dilation=self._dilation_rate,
            )
        elif self._asymmetric:
            # fmt: off
            conv = nn.Sequential(OrderedDict([
                ("conv1", nn.Conv2d(self._reduced_channels, self._reduced_channels, [5, 1], padding=(2, 0),
                                   bias=self._use_bias)),
                ("conv2", nn.Conv2d(self._reduced_channels, self._reduced_channels, [1, 5], padding=(0, 2)))
            ]))
            # fmt: on

        # fmt: off
        self._middle_block = nn.Sequential(OrderedDict([
            ("conv", conv),
            ("batch_norm", nn.BatchNorm2d(self._reduced_channels, 1e-3)),
            ("block_activation", self._block_activation_func(self._reduced_channels))
        ]))
        # fmt: on

        # fmt: off
        self._block1x1_2 = nn.Sequential(OrderedDict([
            ("conv1x1_2", nn.Conv2d(self._reduced_channels, self._output_channels, 1, bias=self._use_bias)),
            ("batch_norm2", nn.BatchNorm2d(self._output_channels, 1e-3)),
            ("block_activation2", self._block_activation_func(self._output_channels))
        ]))
        # fmt: on

        self._dropout = nn.Dropout2d(regularlizer_prob)

        self._convolution_branch = nn.Sequential(
            OrderedDict(
                [
                    ("block1x1_1", self._block1x1_1),
                    ("middle_block", self._middle_block),
                    ("block1x1_2", self._block1x1_2),
                    ("regularizer", self._dropout),
                ]
            )
        )

    def forward(self, x, pooling_indices=None):
        pooling_branch_output = x
        batch_size, _, height, width = x.size()

        if self._downsampling:
            pooling_branch_output, indices = self._pool(pooling_branch_output)

            if self._output_channels != self._input_channels:
                pad = Variable(
                    torch.Tensor(
                        batch_size, self._output_channels - self._input_channels, height // 2, width // 2,
                    ).zero_(),
                    requires_grad=False,
                )

                if torch.cuda.is_available:
                    pad = pad.cuda(0)

                pooling_branch_output = torch.cat((pooling_branch_output, pad), 1)
        elif self._upsampling:
            pooling_branch_output = self._unpool(self._conv_before_unpool(pooling_branch_output), pooling_indices)

        output = pooling_branch_output + self._convolution_branch(x)
        output = self._block_activation_func(channels=self._output_channels)(output)

        if self._downsampling:
            return output, indices

        return output


class Encoder(nn.Module):
    def __init__(self, num_classes, img_size, encoder_only=True):
        super(Encoder, self).__init__()

        self._num_classes = num_classes
        # img_size -> (height, width)
        self._img_size = img_size
        self._encoder_only = encoder_only

        # fmt: off
        self._layers_dict = OrderedDict([
            ("initial", InitialBlock()),
            ("bottleneck_1_0", BottleNeck(16, 64, regularlizer_prob=0.01, downsampling=True)),
            ("bottleneck_1_1", BottleNeck(64, 64, regularlizer_prob=0.01)),
            ("bottleneck_1_2", BottleNeck(64, 64, regularlizer_prob=0.01)),
            ("bottleneck_1_3", BottleNeck(64, 64, regularlizer_prob=0.01)),
            ("bottleneck_1_4", BottleNeck(64, 64, regularlizer_prob=0.01)),
            ("bottleneck_2_0", BottleNeck(64, 128, downsampling=True)),
            ("bottleneck_2_1", BottleNeck(128, 128)),
            ("bottleneck_2_2", BottleNeck(128, 128, dilated=True, dilation_rate=2)),
            ("bottleneck_2_3", BottleNeck(128, 128, asymmetric=True)),
            ("bottleneck_2_4", BottleNeck(128, 128, dilated=True, dilation_rate=4)),
            ("bottleneck_2_5", BottleNeck(128, 128)),
            ("bottleneck_2_6", BottleNeck(128, 128, dilated=True, dilation_rate=8)),
            ("bottleneck_2_7", BottleNeck(128, 128, asymmetric=True)),
            ("bottleneck_2_8", BottleNeck(128, 128, dilated=True, dilation_rate=16)),
            ("bottleneck_3_1", BottleNeck(128, 128)),
            ("bottleneck_3_2", BottleNeck(128, 128, dilated=True, dilation_rate=2)),
            ("bottleneck_3_3", BottleNeck(128, 128, asymmetric=True)),
            ("bottleneck_3_4", BottleNeck(128, 128, dilated=True, dilation_rate=4)),
            ("bottleneck_3_5", BottleNeck(128, 128)),
            ("bottleneck_3_6", BottleNeck(128, 128, dilated=True, dilation_rate=8)),
            ("bottleneck_3_7", BottleNeck(128, 128, asymmetric=True)),
            ("bottleneck_3_8", BottleNeck(128, 128, dilated=True, dilation_rate=16)),
            ("classifier", nn.Conv2d(128, self._num_classes, 1))
        ])
        # fmt: on

    def forward(self, x):
        output = x
        pooling_stack = []

        for key, layer in self._layers_dict.items():
            print(key)
            if hasattr(layer, "downsampling") and layer.downsampling:
                output, pooling_indices = layer(output)
                pooling_stack.append(pooling_indices)
            else:
                output = layer(output)

        if self._encoder_only:
            output = F.upsample(output, self._img_size, None, "bilinear")

        return output, pooling_stack


class Decoder(nn.Module):
    def __init__(self, num_classes, img_size):
        super(Decoder, self).__init__()

        self._num_classes = num_classes
        # img_size -> (height, width)
        self._img_size = img_size

        # fmt: off
        self._layers_dict = OrderedDict([
            ("bottleneck_4_0", BottleNeck(128, 64, upsampling=True, use_relu=True)),
            ("bottleneck_4_1", BottleNeck(64, 64, use_relu=True)),
            ("bottleneck_4_2", BottleNeck(64, 64, use_relu=True)),
            ("bottleneck_5_0", BottleNeck(64, 16, upsampling=True, use_relu=True)),
            ("bottleneck_5_1", BottleNeck(16, 16, use_relu=True)),
            ("fullconv", nn.ConvTranspose2d(16, self._num_classes, 2, stride=2)),
        ])
        # fmt: on

    def forward(self, x, pooling_stack):
        output = x

        for _, layer in self._layers_dict.items():
            if hasattr(layer, "upsampling") and layer.upsampling:
                pooling_indices = pooling_stack.pop()
                output = layer(output, pooling_indices)
            else:
                output = layer(output)

        return output


class Enet(nn.Module):
    def __init__(self, num_classes, img_size, encoder_only=True):
        super(Enet, self).__init__()

        self._num_classes = num_classes
        self._img_size = img_size
        self._encoder_only = encoder_only

        self._encoder = Encoder(self._num_classes, self._img_size, self._encoder_only)
        self._decoder = Decoder(self._num_classes, img_size)

    def forward(self, x):
        output = x
        output, pooling_stack = self._encoder(output)

        if not self._encoder_only:
            output = self._decoder(output, pooling_stack)(output)

        return output
