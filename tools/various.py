#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, TypeVar, \
    MutableSequence, MutableMapping, List, \
    Dict, Type, Union
from argparse import ArgumentParser
from sys import stdout

from torch import Tensor
from torch.nn import Module
from loguru import logger

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['CheckAllNone', 'get_argument_parser',
           'apply_layer', 'init_loggers']


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
                [['--model', '-m'],  # 1st argument
                 {'type': str,
                  'required': True,
                  'help': 'The name of the model to use. '
                          'Accepted values are: `baseline`, '
                          '`baseline_dilated`, `dessed`, '
                          '`dessed_dilated`'}],
                # ----------------------------------------
                [['--config-file', '-c'],  # 1st argument
                 {'type': str,
                  'required': True,
                  'help': 'The name (without extension) '
                          'of the YAML file with the settings.'}],
                # ---------------------------------
                [['--file-dir', '-d'],  # 2nd argument
                 {'type': str,
                  'default': 'settings',
                  'help': 'Directory that holds the settings file (default: `settings`).'}],
                # ---------------------------------
                [['--file-ext', '-e'],  # 3rd argument
                 {'type': str,
                  'default': '.yaml',
                  'help': 'Extension of the settings file (default: `.yaml`).'}],
                # ----------------------------------------
                [['--verbose', '-v'],  # 4th argument
                 {'default': True,
                  'action': 'store_true',
                  'help': 'Be verbose flag (default True).'}]]

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


def init_loggers(verbose: bool) \
        -> None:
    """Initializes the logging process.

    :param verbose: Be verbose?
    :type verbose: bool
    """
    logger.remove()

    for indent in range(3):
        log_string = '{level} | [{time}] {name} -- {space}{message}'.format(
            level='{level}',
            time='{time:HH:mm:ss}',
            name='{name}',
            message='{message}',
            space=' ' * (indent*2))
        logger.add(
            stdout,
            format=log_string,
            level='INFO',
            filter=lambda record, i=indent: record['extra']['indent'] == i)

    if not verbose:
        logger.disable()

# EOF
