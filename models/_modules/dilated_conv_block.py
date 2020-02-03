#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Tuple

from torch import Tensor
from torch.nn import Module, Conv2d, BatchNorm2d, ReLU

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['DilatedConvBLock']


class DilatedConvBLock(Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]],
                 dilation: Union[int, Tuple[int, int]]) \
            -> None:
        """Dilated convolution block.

        :param in_channels: Amount of input channels.
        :type in_channels: int
        :param out_channels: Amount of output channels.
        :type out_channels: int
        :param kernel_size: Kernel shape.
        :type kernel_size: int|(int, int)
        :param stride: Stride shape.
        :type stride: int|(int, int)
        :param padding: Padding shape.
        :type padding: int|(int, int)
        :param dilation: Dilation shape.
        :type dilation: int|(int, int)
        """
        super().__init__()

        self.cnn = Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=kernel_size,
                          stride=stride, padding=padding,
                          dilation=dilation, bias=True)

        self.batch_norm = BatchNorm2d(
            num_features=out_channels)

        self.non_linearity = ReLU()

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the dilated\
        convolution block.

        :param x: Input.
        :type x: torch.Tensor
        :return: Output.
        :rtype: torch.Tensor
        """
        return self.batch_norm(
            self.non_linearity(self.cnn(x)))

# EOF
