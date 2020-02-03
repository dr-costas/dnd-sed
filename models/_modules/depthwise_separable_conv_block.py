#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import reduce
from typing import Union, Tuple, MutableSequence, List, Optional

from torch import Tensor
from torch.nn import Module, Conv2d, BatchNorm2d, LeakyReLU

from tools.various import apply_layer

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['DepthWiseSeparableConvBlock']


class DepthWiseSeparableConvBlock(Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int], MutableSequence[int]],
                 stride: Optional[int] = 1,
                 padding: Optional[int] = 0,
                 dilation: Optional[int] = 1,
                 bias: Optional[bool] = True,
                 padding_mode: Optional[str] = 'zeros',
                 inner_kernel_size: Optional[Union[int, Tuple[int, int], MutableSequence[int]]] = 1,
                 inner_stride: Optional[int] = 1,
                 inner_padding: Optional[int] = 0) \
            -> None:
        """Depthwise separable 2D Convolution.

        :param in_channels: Input channels.
        :type in_channels: int
        :param out_channels: Output channels.
        :type out_channels: int
        :param kernel_size: Kernel shape/size.
        :type kernel_size: int|tuple|list
        :param stride: Stride.
        :type stride: int|tuple|list
        :param padding: Padding.
        :type padding: int|tuple|list
        :param dilation: Dilation.
        :type dilation: int
        :param bias: Bias.
        :type bias: bool
        :param padding_mode: Padding mode.
        :type padding_mode: str
        :param inner_kernel_size: Kernel shape/size of the second convolution.
        :type inner_kernel_size: int|tuple|list
        :param inner_stride: Inner stride.
        :type inner_stride: int|tuple|list
        :param inner_padding: Inner padding.
        :type inner_padding: int|tuple|list
        """
        super(DepthWiseSeparableConvBlock, self).__init__()

        self.depth_wise_conv: Module = Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            dilation=dilation, groups=in_channels, bias=bias,
            padding_mode=padding_mode)

        self.non_linearity: Module = LeakyReLU()

        self.batch_norm: Module = BatchNorm2d(out_channels)

        self.point_wise: Module = Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=inner_kernel_size, stride=inner_stride,
            padding=inner_padding, dilation=1,
            groups=1, bias=bias, padding_mode=padding_mode)

        self.layers: List[Module] = [
            self.depth_wise_conv,
            self.non_linearity,
            self.batch_norm,
            self.point_wise]

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the module.

        :param x: Input tensor.
        :type x: torch.Tensor
        :return: Output tensor.
        :rtype: torch.Tensor
        """
        return reduce(apply_layer, self.layers, x)

# EOF
