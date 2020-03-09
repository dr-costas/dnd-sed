#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from experiments.experiment_process import do_process
from tools.file_io import load_settings_file
from tools.various import get_argument_parser
from tools.printing import InformAboutProcess, date_and_time

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = []


def main():
    args = get_argument_parser().parse_args()
    model_to_use = args.model
    config_file = args.config_file
    file_ext = args.file_ext
    settings_dir = args.file_dir

    date_and_time()
    with InformAboutProcess('Loading settings'):
        settings = load_settings_file(
            file_name=Path(f'{config_file}{file_ext}'),
            settings_dir=Path(settings_dir))

    do_process(
        settings_path=None,
        settings=settings,
        model_to_use=model_to_use)


if __name__ == '__main__':
    main()

# EOF
