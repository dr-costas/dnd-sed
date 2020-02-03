#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Optional, Union, Tuple, \
    MutableMapping, MutableSequence
from functools import partial
from datetime import datetime
from contextlib import ContextDecorator

from torch.nn import Module
from torch.utils.data import DataLoader

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['cmd_msg', 'date_and_time', 'device_info',
           'InformAboutProcess', 'nb_examples', 'nb_parameters',
           'results_evaluation', 'results_training']


_time_f_spec = '7.2'
_acc_f_spec = '6.2'
_loss_f_spec = '7.3'
_epoch_f_spec = '4'


def cmd_msg(the_msg: str,
            start: Optional[str] = '-- ',
            end: Optional[str] = '\n',
            flush: Optional[bool] = True,
            decorate_prv: Optional[Union[Tuple[str, ], str, None]] = None,
            decorate_nxt: Optional[Union[Tuple[str, ], str, None]] = None) \
        -> None:
    """Prints a message.

    :param the_msg: The message.
    :type the_msg: str
    :param start: Starting decoration.
    :type start: str
    :param end: Ending character.
    :type end: str
    :param flush: Flush buffer now? Default to True
    :type flush: bool, optional
    :param decorate_prv: Decoration for previous line.\
                         If argument is tuple or strings,\
                         then its element in the tuple is\
                         printed on one previous line.\
                         Default to None
    :type decorate_prv: tuple(str)|str|None, optional
    :param decorate_nxt: Decoration for next line.\
                         If argument is tuple or strings,\
                         then its element in the tuple is\
                         printed on one previous line.\
                         Default to None
    :type decorate_nxt: tuple(str)|str|None, optional
    """
    msg_len = len(str(the_msg))
    print_w_n = partial(print, end='\n', flush=flush)

    if decorate_prv is not None:
        dec = [decorate_prv] if type(decorate_prv) == str else decorate_prv
        [print_w_n(f'{start}{d * msg_len}') for d in dec]

    msg_end = end if decorate_nxt is None else '\n'
    print(f'{start}{the_msg}', end=msg_end, flush=flush)

    if decorate_nxt is not None:
        dec = [decorate_nxt] if type(decorate_nxt) == str else decorate_nxt
        [print_w_n(f'{start}{d * msg_len}') for d in dec[:-1]]
        print(f'{start}{dec[-1] * msg_len}', end=end, flush=flush)


def date_and_time(**cmd_msg_kwargs: MutableMapping[str, Union[str, Tuple[str, ], bool, None]]) \
        -> None:
    """Prints the date and time of `now`. As if it were `now`...

    Just call this function to print the current time and date.
    :param cmd_msg_kwargs: Keyword arguments for the :func:`cmd_msg`.
    :type cmd_msg_kwargs: dict
    """
    cmd_msg(datetime.now().strftime('%Y-%m-%d %H:%M'), **cmd_msg_kwargs)


def device_info(the_device: str) \
        -> None:
    """Prints an informative message about the device that we are using.

    :param the_device: The device.
    :type the_device: str
    """
    from torch.cuda import get_device_name, current_device
    from platform import processor
    actual_device = get_device_name(current_device()) \
        if the_device.startswith('cuda') else processor()
    cmd_msg(f'Using device: `{actual_device}`.')


class InformAboutProcess(ContextDecorator):
    def __init__(self,
                 starting_msg: str,
                 ending_msg: Optional[str] = 'done',
                 start: Optional[str] = '-- ',
                 end: Optional[str] = '\n') \
            -> None:
        """Context manager and decorator for informing about a process.

        :param starting_msg: The starting message, printed before\
                             the process starts.
        :type starting_msg: str
        :param ending_msg: The ending message, printed after process\
                           ends. Default 'done'.
        :type ending_msg: str, optional
        :param start: Starting decorator for the string to be\
                      printed. Default to '-- '
        :type start: str, optional
        :param end: Ending decorator for the string to be\
                    printed. Default to '\n'.
        :type end: str, optional
        """
        super(InformAboutProcess, self).__init__()
        self.starting_msg = starting_msg
        self.ending_msg = ending_msg
        self.start_dec = start
        self.end_dec = end

    def __enter__(self) \
            -> None:
        """Prints message when entering the context.
        """
        cmd_msg(f'{self.starting_msg}... ', start=self.start_dec, end='')

    def __exit__(self,
                 *exc_type) \
            -> bool:
        """Prints message when exiting the context.
        """
        cmd_msg(f'{self.ending_msg}.', start='', end=self.end_dec)
        return False


def nb_examples(data: MutableSequence[DataLoader],
                cases: MutableSequence[str],
                b_size: int) \
        -> None:
    """Prints amount of examples for each case.

    :param data: Data loaders to use.
    :type data: list[torch.utils.data.DataLoader]
    :param cases: Cases of the data splits (e.g. training).
    :type cases: list[str]
    :param b_size: Batch size.
    :type b_size: int
    """
    m_s = '{} examples/batches'
    len_m = max([len(m_s.format(c)) for c in cases])

    endings = ['\n'] * (len(data) - 1)
    endings.append('\n\n')

    [cmd_msg(f'{m_s.format(t_v):<{len_m}}: '
             f'{len(d) * b_size:5d} /{len(d):5d}',
             end=e_s)
        for t_v, d, e_s in zip(cases, data, endings)]


def nb_parameters(module: Module, module_name: str = 'Module'):
    """Prints the amount of parameters in a module.

    :param module: Module to use.
    :type module: torch.nn.Module
    :param module_name: Name of the module, defaults Module.
    :type module_name: str, optional
    """
    cmd_msg(f'{module_name} has '
            f'{sum([i.numel() for i in module.parameters()])} '
            f'parameters.')


def results_evaluation(f1_score: float,
                       er_score: float,
                       time_elapsed: float) -> None:
    """Prints the output of the testing process.

    :param f1_score: F1 score.
    :type f1_score: float
    :param er_score: Error rate.
    :type er_score: float
    :param time_elapsed: Elapsed time for the epoch.
    :type time_elapsed: float
    """
    the_msg = f'F1:{f1_score:{_acc_f_spec}f} | ' \
              f'ER:{er_score:{_acc_f_spec}f} | ' \
              f'Time:{time_elapsed:{_time_f_spec}f}'

    cmd_msg(the_msg, start='  -- ')


def results_training(epoch: int,
                     training_loss: float,
                     validation_loss: Union[float, None],
                     training_f1: float,
                     training_er: float,
                     validation_f1: Union[float, None],
                     validation_er: Union[float, None],
                     time_elapsed: float):
    """Prints the results of the pre-training step to console.

    :param epoch: Epoch.
    :type epoch: int
    :param training_loss: Loss of the training data.
    :type training_loss: float
    :param validation_loss: Loss of the validation data.
    :type validation_loss: float | None
    :param training_f1: F1 score for the training data.
    :type training_f1: float
    :param training_er: Error rate for the training data.
    :type training_er: float
    :param validation_f1: F1 score for the validation data.
    :type validation_f1: float | None
    :param validation_er: Error rate for the validation data.
    :type validation_er: float | None
    :param time_elapsed: Time elapsed for the epoch.
    :type time_elapsed: float
    """
    validation_loss = 'None' if validation_loss is None else validation_loss
    validation_f1 = 'None' if validation_f1 is None else validation_f1
    validation_er = 'None' if validation_er is None else validation_er

    the_msg = f'Epoch:{epoch:{_epoch_f_spec}d} | ' \
              f'Loss (tr/va):{training_loss:{_loss_f_spec}f}/{validation_loss:{_loss_f_spec}f} | ' \
              f'F1 (tr/va):{training_f1:{_acc_f_spec}f}/{validation_f1:{_acc_f_spec}f} | ' \
              f'ER (tr/va):{training_er:{_acc_f_spec}f}/{validation_er:{_acc_f_spec}f} | ' \
              f'Time:{time_elapsed:{_time_f_spec}f} sec.'

    cmd_msg(the_msg, start='  -- ')


# EOF
