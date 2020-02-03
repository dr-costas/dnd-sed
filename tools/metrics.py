#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Tuple, Union

import torch

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['f1_per_frame', 'error_rate_per_frame']


_eps: float = torch.finfo(torch.float32).eps


def f1_per_frame(y_hat: torch.Tensor,
                 y_true: torch.Tensor) \
        -> torch.Tensor:
    """Gets the average per frame F1 score, based on\
    TP, FP, and FN, calculated from the `y_hat`\
    predictions and `y_true` ground truth values.

    :param y_hat: Predictions
    :type y_hat: torch.Tensor
    :param y_true: Ground truth values
    :type y_true: torch.Tensor
    :return: F1 score per frame
    :rtype: torch.Tensor
    """
    tp, _, fp, fn = _tp_tf_fp_fn(
        y_hat=y_hat, y_true=y_true,
        dim_sum=None)

    tp = tp.sum()
    fp = fp.sum()
    fn = fn.sum()
    the_f1 = _f1(tp=tp, fp=fp, fn=fn)

    return the_f1


def error_rate_per_frame(y_hat: torch.Tensor,
                         y_true: torch.Tensor) \
        -> torch.Tensor:
    """Calculates the error rate based on FN and FP,
    for one second.

    :param y_hat: Predictions.
    :type y_hat: torch.Tensor
    :param y_true: Ground truth.
    :type y_true: torch.Tensor
    :return: Error rate.
    :rtype: torch.Tensor
    """
    _, __, fp, fn = _tp_tf_fp_fn(
        y_hat=y_hat, y_true=y_true,
        dim_sum=-1)

    s = fn.min(fp).sum()
    d = fn.sub(fp).clamp_min(0).sum()
    i = fp.sub(fn).clamp_min(0).sum()
    n = y_true.sum() + _eps

    return (s + d + i)/n


def _f1(tp: torch.Tensor,
        fp: torch.Tensor,
        fn: torch.Tensor) \
        -> torch.Tensor:
    """Gets the F1 score from the TP, FP, and FN.

    :param tp: TP
    :type tp: torch.Tensor
    :param fp: FP
    :type fp: torch.Tensor
    :param fn: FN
    :type fn: torch.Tensor
    :return: F1 score
    :rtype: torch.Tensor
    """
    if all([m.sum().item() == 0 for m in [tp, fp, fn]]):
        return torch.zeros(1).to(tp.device)

    f1_nominator = tp.mul(2)
    f1_denominator = tp.mul(2).add(fn).add(fp)

    return f1_nominator.div(f1_denominator + _eps)


def _tp_tf_fp_fn(y_hat: torch.Tensor,
                 y_true: torch.Tensor,
                 dim_sum: Union[int, None]) \
        -> Tuple[torch.Tensor, torch.Tensor,
                 torch.Tensor, torch.Tensor]:
    """Gets the true positive (TP), true negative (TN),\
    false positive (FP), and false negative (FN).

    :param y_hat: Predictions
    :type y_hat: torch.Tensor
    :param y_true: Ground truth values
    :type y_true: torch.Tensor
    :param dim_sum: Dimension to sum TP, TN, FP, and FN. If\
                    it is None, then the default behaviour from\
                    PyTorch`s sum is assumed.
    :type dim_sum: int|None
    :return: TP, TN, FP, FN.
    :rtype: (torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor)
    """
    y_hat_positive = y_hat.ge(0.5)
    y_hat_negative = y_hat.lt(0.5)

    y_true_positive = y_true.eq(1.)
    y_true_negative = y_true.eq(0.)

    tp = y_hat_positive.mul(y_true_positive).view(
        -1, y_hat_positive.size()[-1]
    ).float()

    tn = y_hat_negative.mul(y_true_negative).view(
        -1, y_hat_positive.size()[-1]
    ).float()

    fp = y_hat_positive.mul(y_true_negative).view(
        -1, y_hat_positive.size()[-1]
    ).float()

    fn = y_hat_negative.mul(y_true_positive).view(
        -1, y_hat_positive.size()[-1]
    ).float()

    if dim_sum is not None:
        tp = tp.sum(dim=dim_sum)
        tn = tn.sum(dim=dim_sum)
        fp = fp.sum(dim=dim_sum)
        fn = fn.sum(dim=dim_sum)

    return tp, tn, fp, fn

# EOF
