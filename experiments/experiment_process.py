#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union, MutableMapping
from pathlib import Path

from tools.file_io import load_settings_file
from tools.printing import cmd_msg, date_and_time
from tools.various import get_argument_parser, CheckAllNone
from models import CRNN, DESSED, DESSEDDilated, BaselineDilated
from ._processes import experiment

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['do_process']


@CheckAllNone()
def do_process(settings_path: Optional[Union[Path, None]] = None,
               settings: Optional[Union[MutableMapping, None]] = None,
               model_to_use: Optional[Union[str, None]] = None) \
        -> None:
    """The process of the baseline experiment.

    :param settings_path: Path for the settings. Defaults to None.
    :type settings_path: str|None, optional
    :param settings: Settings. Defaults to None.
    :type settings: dict|None, optional
    :param model_to_use: Model to use. Defaults to None.
    :type model_to_use: str, optional
    """
    if settings_path is not None:
        settings = load_settings_file(Path(f'{settings_path}.yaml'))

    if model_to_use == 'baseline':
        msg = 'Baseline experiment'
        model = CRNN
    elif model_to_use == 'baseline_dilated':
        msg = 'Baseline with dilated convolutions experiment'
        model = BaselineDilated
    elif model_to_use == 'dessed':
        msg = 'Depth-wise separable with RNN experiment'
        model = DESSED
    elif model_to_use == 'dessed_dilated':
        msg = 'Depth-wise separable with dilated convolutions experiment'
        model = DESSEDDilated
    else:
        raise AttributeError(f'Unrecognized model `{model_to_use}`. '
                             f'Accepted model names are: `baseline`, '
                             '`baseline_dilated`, `dessed`, '
                             '`dessed_dilated`.')

    cmd_msg(msg, start='\n-- ')
    model_settings = settings[model_to_use]

    cmd_msg('Starting experiment', end='\n\n')
    experiment(settings=settings,
               model_settings=model_settings,
               model_class=model)


def main():
    date_and_time()

    arg_parser = get_argument_parser()
    args = arg_parser.parse_args()

    do_process(args.config_file)


if __name__ == '__main__':
    main()

# EOF
