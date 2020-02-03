#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from datetime import datetime
from contextlib import ContextDecorator

from torch.nn import Module

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['cmd_msg', 'date_and_time', 'device_info',
           'InformAboutProcess', 'nb_examples', 'nb_parameters',
           'results_evaluation', 'results_training']


_time_f_spec = '7.2'
_acc_f_spec = '6.2'
_loss_f_spec = '7.3'
_epoch_f_spec = '4'
_s_data = '{m:<{len_m}}: {d1:5d} /{d2:5d}'


def cmd_msg(the_msg, start='-- ', end='\n', flush=True,
            decorate_prv=None, decorate_nxt=None):
    """Prints a message.

    :param the_msg: The message.
    :type the_msg: str
    :param start: Starting decoration.
    :type start: str
    :param end: Ending character.
    :type end: str
    :param flush: Flush buffer now?
    :type flush: bool
    :param decorate_prv: Decoration for previous line.\
                         If argument is tuple or strings,\
                         then its element in the tuple is\
                         printed on one previous line.
    :type decorate_prv: tuple(str)|str|None
    :param decorate_nxt: Decoration for next line.\
                         If argument is tuple or strings,\
                         then its element in the tuple is\
                         printed on one previous line.
    :type decorate_nxt: tuple(str)|str|None
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


def date_and_time(**cmd_msg_kwargs):
    """Prints the date and time of `now`. As if it were `now`...

    Just call this function to print the current time and date.
    :param cmd_msg_kwargs: Keyword arguments for the :func:`cmd_msg`.
    :type cmd_msg_kwargs: dict
    """
    cmd_msg(datetime.now().strftime('%Y-%m-%d %H:%M'), **cmd_msg_kwargs)


def device_info(the_device):
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
    def __init__(self, starting_msg, ending_msg='done', start='-- ', end='\n'):
        """Context manager and decorator for informing about a process.

        :param starting_msg: The starting message, printed before the process starts.
        :type starting_msg: str
        :param ending_msg: The ending message, printed after process ends.
        :type ending_msg: str
        :param start: Starting decorator for the string to be printed.
        :type start: str
        :param end: Ending decorator for the string to be printed.
        :type end: str
        """
        super(InformAboutProcess, self).__init__()
        self.starting_msg = starting_msg
        self.ending_msg = ending_msg
        self.start_dec = start
        self.end_dec = end

    def __enter__(self):
        cmd_msg(f'{self.starting_msg}... ', start=self.start_dec, end='')

    def __exit__(self, *exc_type):
        cmd_msg(f'{self.ending_msg}.', start='', end=self.end_dec)
        return False


def nb_examples(data, cases, b_size):
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

    [cmd_msg(_s_data.format(
        m=m_s.format(t_v), d1=len(d) * b_size,
        d2=len(d), len_m=len_m), end=e_s)
        for t_v, d, e_s in zip(cases, data, endings)]


def nb_parameters(module: Module, module_name: str = 'Module'):
    """Prints the amount of parameters in a module.

    :param module: Module to use.
    :type module: torch.nn.Module
    :param module_name: Name of the module, defaults Module
    :type module_name: str, optional
    """
    cmd_msg(f'{module_name} has {sum([i.numel() for i in module.parameters()])} parameters.')


def results_evaluation(f1_score, er_score, time_elapsed):
    """Prints the output of the testing process.

    :param f1_score: The F1 score.
    :type f1_score: float
    :param er_score: The error rate.
    :type er_score: float
    :param time_elapsed: The elapsed time for the epoch.
    :type time_elapsed: float
    """
    the_msg = f'F1:{f1_score:{_acc_f_spec}f} | ' \
              f'ER:{er_score:{_acc_f_spec}f} | ' \
              f'Time:{time_elapsed:{_time_f_spec}f}'

    cmd_msg(the_msg, start='  -- ')


def results_training(epoch, training_loss, validation_loss,
                     training_f1, training_er,
                     validation_f1, validation_er,
                     time_elapsed):
    """Prints the results of the pre-training step to console.

    :param epoch: The epoch.
    :type epoch: int
    :param training_loss: The loss of the training data.
    :type training_loss: float
    :param validation_loss: The loss of the validation data.
    :type validation_loss: float | None
    :param training_f1: The F1 score for the training data.
    :type training_f1: float
    :param training_er: The error rate for the training data.
    :type training_er: float
    :param validation_f1: The F1 score for the validation data.
    :type validation_f1: float | None
    :param validation_er: The error rate for the validation data.
    :type validation_er: float | None
    :param time_elapsed: The time elapsed for the epoch.
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
