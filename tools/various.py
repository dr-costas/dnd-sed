#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, TypeVar, \
    MutableSequence, MutableMapping, List, \
    Dict, Type, Union
from argparse import ArgumentParser

from torch import Tensor
from torch.nn import Module

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['CheckAllNone', 'get_argument_parser', 'apply_layer']


T = TypeVar('T')


class CheckAllNone(object):
    """Decorator to assure that at least one argument is\
    not None.
    """
    def __init__(self) \
            -> None:
        super(CheckAllNone, self).__init__()
        self.fn = None

    def _decorated(self,
                   *args: MutableSequence,
                   **kwargs: MutableMapping) \
            -> T:
        all_args = list(*args)
        all_args.extend(kwargs.values())

        if all(an_arg is None for an_arg in all_args):
            raise AssertionError(
                'Provide at least one not None argument.')
        return self.fn(*args, **kwargs)

    def __call__(self,
                 fn: Callable) \
            -> Callable:
        self.fn = fn
        return self._decorated


def get_argument_parser() \
        -> ArgumentParser:
    """Creates and returns the ArgumentParser for this project.

    :return: Argument parser.
    :rtype: argparse.ArgumentParser
    """
    the_args: List[List[Union[List[str], Dict[str, Union[Type, str]]]]] = [
                # ----------------------------------------
                [['--config-file', '-c'],  # 1st argument
                 {'type': str,
                  'help': 'The name (without extension) '
                          'of the YAML file with the settings.'}],
                # ----------------------------------------
                [['--model', '-m'],  # 2nd argument
                 {'type': str,
                  'help': 'The name (without extension) '
                          'of the YAML file with the settings.'}],
                # ----------------------------------------
                [['--job-id', '-j'],  # 3rd argument
                 {'type': str,
                  'default': str(0),
                  'help': 'The current job id on SLURM.'}]]

    arg_parser = ArgumentParser()
    [arg_parser.add_argument(*i[0], **i[1]) for i in the_args]

    return arg_parser


def apply_layer(layer_input: Tensor,
                layer: Module) \
        -> Tensor:
    """Small aux function to speed up reduce operation.

    :param layer_input: Input to the layer.
    :type layer_input: torch.Tensor
    :param layer: Layer.
    :type layer: torch.nn.Module
    :return: Output of the layer.
    :rtype: torch.Tensor
    """
    return layer(layer_input)

# EOF
