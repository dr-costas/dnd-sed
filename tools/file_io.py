#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
import yaml
import numpy as np

from tools import yaml_loader

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = [
    'load_numpy_object',
    'load_yaml_file',
    'load_settings_file'
]


def load_numpy_object(f_name):
    """Loads and returns a numpy object.

    :param f_name: Path of the object.
    :type f_name: pathlib.Path
    :return: Numpy object.
    :rtype: numpy.ndarray
    """
    return np.load(str(f_name), allow_pickle=True)


def load_settings_file(file_name, settings_dir=pathlib.Path('settings')):
    """Reads and returns the contents of a YAML settings file.

    :param file_name: Name of the settings file.
    :type file_name: pathlib.Path
    :param settings_dir: Directory with the settings files.
    :type settings_dir: pathlib.Path
    :return: Contents of the YAML settings file.
    :rtype: dict
    """
    settings_file_path = settings_dir.joinpath(file_name.with_suffix('.yaml'))
    return load_yaml_file(settings_file_path)


def load_yaml_file(file_path):
    """Reads and returns the contents of a YAML file.

    :param file_path: Path to the YAML file.
    :type file_path: pathlib.Path
    :return: Contents of the YAML file.
    :rtype: dict
    """
    with file_path.open('r') as f:
        return yaml.load(f, Loader=yaml_loader.YAMLLoader)

# EOF
