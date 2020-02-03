#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Tuple, List
from functools import reduce

from torch.nn import Module, Sequential, ReLU, \
    BatchNorm2d, MaxPool2d, Dropout2d

from tools.various import apply_layer
from .depthwise_separable_conv_block import DepthWiseSeparableConvBlock

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['DepthWiseSeparableDNN']


class DepthWiseSeparableDNN(Module):

    def __init__(self,
                 cnn_channels: int,
                 cnn_dropout: float,
                 inner_kernel_size: Union[int, Tuple[int, int]],
                 inner_padding: Union[int, Tuple[int, int]]) \
            -> None:
        """Depthwise separable blocks.

        :param cnn_channels: Amount of output CNN channels. For first\
                             CNN in the block is considered equal to 1.
        :type cnn_channels: int
        :param cnn_dropout: Dropout to apply.
        :type cnn_dropout: float
        :param inner_kernel_size: Kernel shape to use.
        :type inner_kernel_size: (int, int)|int
        :param inner_padding: Padding to use.
        :type inner_padding: (int, int)|int
        """
        super().__init__()

        self.layer_1: Module = Sequential(
            DepthWiseSeparableConvBlock(
                in_channels=1, out_channels=cnn_channels,
                kernel_size=5, stride=1, padding=2,
                inner_kernel_size=inner_kernel_size,
                inner_padding=inner_padding),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, 5), stride=(1, 5)),
            Dropout2d(cnn_dropout))

        self.layer_2: Module = Sequential(
            DepthWiseSeparableConvBlock(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=5, stride=1, padding=2,
                inner_kernel_size=inner_kernel_size,
                inner_padding=inner_padding),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, 4), stride=(1, 4)),
            Dropout2d(cnn_dropout))

        self.layer_3: Module = Sequential(
            DepthWiseSeparableConvBlock(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=5, stride=1, padding=2,
                inner_kernel_size=inner_kernel_size,
                inner_padding=inner_padding),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            Dropout2d(cnn_dropout))

        self.layers: List[Module] = [
            self.layer_1, self.layer_2, self.layer_3]

    def forward(self, x):
        """The forward pass of the DepthWiseSeparableDNN.

        :param x: Input audio features.
        :type x: torch.Tensor
        :return: Learned representation\
                 by the DepthWiseSeparableDNN.
        :rtype: torch.Tensor
        """
        return reduce(
            apply_layer, self.layers,
            x.unsqueeze(1))

# EOF
