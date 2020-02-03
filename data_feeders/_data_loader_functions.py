#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Dict

from torch.utils.data import DataLoader

from ._tut_sed_synthetic_2016 import TUTSEDSynthetic2016

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['get_tut_sed_data_loader']


def get_tut_sed_data_loader(root_dir: str,
                            split: str,
                            batch_size: int,
                            shuffle: bool,
                            drop_last: bool,
                            input_features_file_name: str,
                            target_values_input_name: str) \
        -> DataLoader:
    """Creates and returns the data loader.

    :param root_dir: The root dir for the dataset.
    :type root_dir: str
    :param split: The split of the data (training, \
                          validation, or testing).
    :type split: str
    :param batch_size: The batch size.
    :type batch_size: int
    :param shuffle: Shuffle the data?
    :type shuffle: bool
    :param drop_last: Drop last examples?
    :type drop_last: bool
    :param input_features_file_name: Input features file name.
    :type input_features_file_name: str
    :param target_values_input_name: Target values file name.
    :type target_values_input_name: str
    :return: Data loader.
    :rtype: torch.utils.data.DataLoader
    """
    data_loader_kwargs: Dict = {
        'root_dir': root_dir,
        'split': split,
        'input_features_file_name': input_features_file_name,
        'target_values_input_name': target_values_input_name}

    dataset = TUTSEDSynthetic2016(**data_loader_kwargs)

    return DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=shuffle if split == 'training' else False,
        drop_last=drop_last)

# EOF
