#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Union, Tuple, List

from torch import Tensor
from torch.nn import Module, Linear

from ._modules import DNN
from ._modules import DilatedConvBLock

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['BaselineDilated']


class BaselineDilated(Module):

    def __init__(self,
                 cnn_channels: int,
                 cnn_dropout: float,
                 dilated_output_channels: int,
                 dilated_kernel_size: Union[int, Tuple[int, int], List[int]],
                 dilated_stride: Union[int, Tuple[int, int], List[int]],
                 dilated_padding: Union[int, Tuple[int, int], List[int]],
                 dilation_shape: Union[int, Tuple[int, int], List[int]],
                 dilated_nb_features: int,
                 nb_classes: int) \
            -> None:
        """Baseline model with dilated convolutions.

        :param cnn_channels: Amount of CNN channels.
        :type cnn_channels: int
        :param cnn_dropout: Dropout to be applied to the CNNs.
        :type cnn_dropout: float
        :param dilated_output_channels: Amount of channels for the\
                                        dilated CNN.
        :type dilated_output_channels: int
        :param dilated_kernel_size: Kernel shape of the dilated CNN.
        :type dilated_kernel_size: int|(int, int)|list[int]
        :param dilated_stride: Stride shape of the dilated CNN.
        :type dilated_stride: int|(int, int)|list[int]
        :param dilated_padding: Padding shape of the dilated CNN.
        :type dilated_padding: int|(int, int)|list[int]
        :param dilation_shape: Dilation shape of the dilated CNN.
        :type dilation_shape: int|(int, int)|list[int]
        :param dilated_nb_features: Amount of features for the batch\
                                    norm after the dilated CNN.
        :type dilated_nb_features: int
        :param nb_classes: Amount of classes to be predicted.
        :type nb_classes: int
        """
        super().__init__()

        self.p_1: List[int] = [0, 3, 2, 1]
        self.p_2: List[int] = [0, 2, 1, 3]

        self.dnn: Module = DNN(cnn_channels=cnn_channels,
                               cnn_dropout=cnn_dropout)

        self.dilated_cnn: Module = DilatedConvBLock(
            in_channels=1, out_channels=dilated_output_channels,
            kernel_size=dilated_kernel_size, stride=dilated_stride,
            padding=dilated_padding, dilation=dilation_shape)

        self.classifier: Module = Linear(
            in_features=dilated_nb_features * dilated_output_channels,
            out_features=nb_classes, bias=True)

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the BaselineDilated model.

        :param x: Input to the BaselineDilated.
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        r_s: List[int] = list(x.size()[:-1]) + [-1]

        out: Tensor = self.dnn(x).permute(
            *self.p_1).contiguous()

        out: Tensor = self.dilated_cnn(out).permute(
            *self.p_2).reshape(*r_s)

        return self.classifier(out)

# EOF
