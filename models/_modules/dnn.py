#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List
from functools import reduce

from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, ReLU, \
    BatchNorm2d, Dropout2d, MaxPool2d

from tools.various import apply_layer

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['DNN']


class DNN(Module):

    def __init__(self,
                 cnn_channels: int,
                 cnn_dropout: float) \
            -> None:
        """Baseline CNN blocks.

        :param cnn_channels: CNN channels.
        :type cnn_channels: int
        :param cnn_dropout: Dropout to apply.
        :type cnn_dropout: float
        """
        super().__init__()

        self.layer_1: Module = Sequential(
            Conv2d(
                in_channels=1, out_channels=cnn_channels,
                kernel_size=5, stride=1, padding=2),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, 5),
                      stride=(1, 5)),
            Dropout2d(cnn_dropout)
        )

        self.layer_2: Module = Sequential(
            Conv2d(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=5, stride=1,
                padding=2),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, 4),
                      stride=(1, 4)),
            Dropout2d(cnn_dropout)
        )

        self.layer_3: Module = Sequential(
            Conv2d(
                in_channels=cnn_channels,
                out_channels=cnn_channels,
                kernel_size=5, stride=1,
                padding=2),
            ReLU(),
            BatchNorm2d(cnn_channels),
            MaxPool2d(kernel_size=(1, 2),
                      stride=(1, 2)),
            Dropout2d(cnn_dropout)
        )

        self.layers: List[Module] = [
            self.layer_1, self.layer_2, self.layer_3]

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the baseline CNN blocks.

        :param x: Input audio features.
        :type x: torch.Tensor
        :return: Learned representation\
                 by the baseline CNN blocks.
        :rtype: torch.Tensor
        """
        return reduce(
            apply_layer, self.layers,
            x.unsqueeze(1))

# EOF
