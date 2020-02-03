#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import zeros, Tensor
from torch.nn import Module, GRUCell, Linear

from ._modules import DepthWiseSeparableDNN

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['DESSED']


class DESSED(Module):

    def __init__(self,
                 cnn_channels: int,
                 cnn_dropout: float,
                 rnn_in_dim: int,
                 rnn_out_dim: int,
                 nb_classes: int) \
            -> None:
        """The DESSED model.

        :param cnn_channels: Amount of CNN channels.
        :type cnn_channels: int
        :param cnn_dropout: Dropout to be applied to the CNNs.
        :type cnn_dropout: float
        :param rnn_in_dim: Input dimensionality of the RNN.
        :type rnn_in_dim: int
        :param rnn_out_dim: Output dimensionality of the RNN.
        :type rnn_out_dim: int
        :param nb_classes: Amount of classes to be predicted.
        :type nb_classes: int
        """
        super().__init__()

        self.rnn_hh_size: int = rnn_out_dim
        self.nb_classes: int = nb_classes

        self.dnn: Module = DepthWiseSeparableDNN(
            cnn_channels=cnn_channels,
            cnn_dropout=cnn_dropout)

        self.rnn: Module = GRUCell(
            input_size=rnn_in_dim,
            hidden_size=self.rnn_hh_size,
            bias=True)

        self.classifier: Module = Linear(
            in_features=self.rnn_hh_size,
            out_features=self.nb_classes,
            bias=True)

    def forward(self,
                x: Tensor) \
            -> Tensor:
        """Forward pass of the DESSED model.

        :param x: Input to the DESSED.
        :type x: torch.Tensor
        :return: Output predictions.
        :rtype: torch.Tensor
        """
        b_size, t_steps, _ = x.size()
        feats: Tensor = self.dnn(x).permute(
            0, 2, 1, 3
        ).reshape(b_size, t_steps, -1)

        h: Tensor = zeros(
            b_size, self.rnn_hh_size
        ).to(x.device)

        outputs: Tensor = zeros(
            b_size, t_steps, self.nb_classes
        ).to(x.device)

        for t_step, h_f in enumerate(feats.permute(1, 0, 2)):
            h: Tensor = self.rnn(h_f, h)
            outputs[:, t_step, :] = self.classifier(h)

        return outputs
# EOF
