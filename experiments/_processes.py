#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Callable, Optional, Tuple, \
    Union, MutableMapping
from time import time
from copy import deepcopy

from torch import no_grad, cat, zeros, Tensor
from torch.optim import optimizer as pt_opt, Adam
from torch.nn import BCEWithLogitsLoss, utils, Module
from torch.cuda import is_available
from torch.utils.data import DataLoader

from tools.metrics import f1_per_frame, error_rate_per_frame
from tools.printing import results_evaluation, results_training, \
    nb_examples, cmd_msg, nb_parameters, device_info, InformAboutProcess
from data_feeders import get_tut_sed_data_loader

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['training', 'testing', 'experiment']


def _sed_epoch(model: Module,
               data_loader: DataLoader,
               objective: Union[Callable, None],
               optimizer: Union[pt_opt.Optimizer, None],
               device: str,
               grad_norm: Optional[float] = 1.) \
        -> Tuple[Module, Tensor, Tensor, Tensor]:
    """Performs a forward pass for the BREACNNModel model.

    :param model: The BREACNNModel model.
    :type model: torch.nn.Module
    :param data_loader: The data loader to be used.
    :type data_loader: torch.utils.data.DataLoader
    :param objective: The objective function to be used.
    :type objective: callable | None
    :param optimizer: The optimizer ot be used.
    :type optimizer: torch.optim.Optimizer | None
    :param device: The device to be used.
    :type device: str
    :param grad_norm: The maximum gradient norm.
    :type grad_norm: float
    :return: The model and the values for the objective and evaluation of a full\
             iteration of the data (objective, f1_score, er_score).
    :rtype: torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor
    """
    epoch_objective_values: Tensor = zeros(len(data_loader)).float()
    values_true, values_hat = [], []

    for e, data in enumerate(data_loader):
        if optimizer is not None:
            optimizer.zero_grad()

        x, y = [i.float().to(device) for i in data]

        y_hat: Tensor = model(x)

        loss = 0.

        if objective is not None:
            loss: Tensor = objective(y_hat, y)
            if optimizer is not None:
                loss.backward()
                if grad_norm > 0:
                    utils.clip_grad_norm_(model.parameters(), grad_norm)
                optimizer.step()
            loss: float = loss.item()

        epoch_objective_values[e] = loss
        values_true.append(y.cpu())
        values_hat.append(y_hat.cpu())

    values_true = cat(values_true, dim=0)
    values_hat = cat(values_hat, dim=0)

    return model, epoch_objective_values, values_true, values_hat


def testing(model: Module,
            data_loader: DataLoader,
            f1_func: Callable,
            er_func: Callable,
            device: str):
    """Tests a model.

    :param model: Model to be tested.
    :type model: torch.nn.Module
    :param data_loader: Data loader to be used.
    :type data_loader: torch.utils.data.DataLoader
    :param f1_func: Function to obtain F1 score.
    :type f1_func: callable
    :param er_func: Function to obtain error rate.
    :type er_func: callable
    :param device: Device to be used.
    :type device: str
    """
    start_time = time()
    model.eval()
    with no_grad():
        _, _, true_values, hat_values = _sed_epoch(
            model=model, data_loader=data_loader,
            objective=None, optimizer=None,
            device=device)

    end_time = time() - start_time

    f1_score = f1_func(hat_values, true_values).mean()
    er_score = er_func(hat_values, true_values).mean()

    results_evaluation(f1_score, er_score, end_time)


def training(model:Module,
             data_loader_training: DataLoader,
             optimizer: pt_opt.Optimizer,
             objective: Callable,
             f1_func: Callable,
             er_func: Callable,
             epochs: int,
             data_loader_validation: DataLoader,
             validation_patience: int,
             device: str,
             grad_norm: float) \
        -> Module:
    """Optimizes a model.

    :param model: Model to optimize.
    :type model: torch.nn.Module
    :param data_loader_training: Data loader to be used with\
                                 the training data.
    :type data_loader_training: torch.utils.data.DataLoader
    :param optimizer: Optimizer to be used.
    :type optimizer: torch.optim.Optimizer
    :param objective: Objective function to be used.
    :type objective: callable
    :param f1_func: Function to calculate the F1 score.
    :type f1_func: callable
    :param er_func: Function to calculate the error rate.
    :type er_func: callable
    :param epochs: Maximum amount of epochs for training.
    :type epochs: int
    :param data_loader_validation:Data loader to be used with\
                                 the validation data.
    :type data_loader_validation: torch.utils.data.DataLoader
    :param validation_patience: Maximum amount of epochs for waiting\
                                for validation score improvement.
    :type validation_patience: int
    :param device: Device to be used.
    :type device: str
    :param grad_norm: Maximum gradient norm.
    :type grad_norm: float
    :return: Optimized model.
    :rtype: torch.nn.Module
    """
    best_model = None
    epochs_waiting = 100
    lowest_epoch_loss = 1e8
    best_model_epoch = -1

    for epoch in range(epochs):
        start_time = time()

        model = model.train()
        model, epoch_tr_loss, true_training, hat_training = _sed_epoch(
            model=model, data_loader=data_loader_training,
            objective=objective, optimizer=optimizer,
            device=device, grad_norm=grad_norm)

        epoch_tr_loss = epoch_tr_loss.mean().item()

        f1_score_training = f1_func(
            hat_training.sigmoid(),
            true_training).mean().item()

        error_rate_training = er_func(
            hat_training.sigmoid(),
            true_training).mean().item()

        model = model.eval()
        with no_grad():
            model, epoch_va_loss, true_validation, hat_validation = _sed_epoch(
                model=model, data_loader=data_loader_validation,
                objective=objective, optimizer=None,
                device=device)

        epoch_va_loss = epoch_va_loss.mean().item()

        f1_score_validation = f1_func(
            hat_validation.sigmoid(),
            true_validation).mean().item()

        error_rate_validation = er_func(
            hat_validation.sigmoid(),
            true_validation).mean().item()

        if epoch_va_loss < lowest_epoch_loss:
            lowest_epoch_loss = epoch_va_loss
            epochs_waiting = 0
            best_model = deepcopy(model.state_dict())
            best_model_epoch = epoch
        else:
            epochs_waiting += 1

        end_time = time() - start_time

        results_training(
            epoch=epoch, training_loss=epoch_tr_loss,
            validation_loss=epoch_va_loss,
            training_f1=f1_score_training,
            training_er=error_rate_training,
            validation_f1=f1_score_validation,
            validation_er=error_rate_validation,
            time_elapsed=end_time)

        if epochs_waiting >= validation_patience:
            cmd_msg(f'Early stopping! Lowest validation loss: {lowest_epoch_loss:7.3f} '
                    f'at epoch: {best_model_epoch:3d}', start='\n-- ', end='\n\n')
            break

    if best_model is not None:
        model.load_state_dict(best_model)

    return model


def experiment(settings: MutableMapping,
               model_settings: MutableMapping,
               model_class: Callable) \
        -> None:
    """Does the experiment with the specified settings and model.

    :param settings: General settings.
    :type settings: dict
    :param model_settings: Model settings.
    :type model_settings: dict
    :param model_class: The class of the model.
    :type model_class: callable
    """
    device = 'cuda' if is_available() else 'cpu'

    with InformAboutProcess('Creating the model'):
        model = model_class(**model_settings)
        model = model.to(device)

    with InformAboutProcess('Creating training data loader'):
        training_data = get_tut_sed_data_loader(
            split='training',
            **settings['data_loader'])

    with InformAboutProcess('Creating validation data loader'):
        validation_data = get_tut_sed_data_loader(
            split='validation',
            **settings['data_loader'])

    with InformAboutProcess('Creating optimizer'):
        optimizer = Adam(model.parameters(),
                         lr=settings['optimizer']['lr'])

    cmd_msg('', start='')

    common_kwargs = {'f1_func': f1_per_frame,
                     'er_func': error_rate_per_frame,
                     'device': device}

    nb_examples(
        [training_data, validation_data],
        ['Training', 'Validation'],
        settings['data_loader']['batch_size'])

    if hasattr(model, 'dnn'):
        nb_parameters(model.dnn, 'DNN')
    if hasattr(model, 'dilated_cnn'):
        nb_parameters(model.dilated_cnn, 'Dilated CNN')
    if hasattr(model, 'rnn'):
        nb_parameters(model.rnn, 'RNN')
    nb_parameters(model.classifier, 'Classifier')
    nb_parameters(model)

    cmd_msg('', start='')
    device_info(device)

    cmd_msg('Starting training', start='\n\n-- ', end='\n\n')

    optimized_model = training(
        model=model, data_loader_training=training_data,
        optimizer=optimizer, objective=BCEWithLogitsLoss(),
        epochs=settings['training']['epochs'],
        data_loader_validation=validation_data,
        validation_patience=settings['training']['validation_patience'],
        grad_norm=settings['training']['grad_norm'], **common_kwargs)

    del training_data
    del validation_data

    with InformAboutProcess('Creating testing data loader'):
        testing_data = get_tut_sed_data_loader(
            split='testing',
            **settings['data_loader'])

    nb_examples([testing_data], ['Testing'],
                settings['data_loader']['batch_size'])

    cmd_msg('Starting testing', start='\n\n-- ', end='\n\n')
    testing(model=optimized_model, data_loader=testing_data,
            **common_kwargs)

    cmd_msg('That\'s all!', start='\n\n-- ', end='\n\n')

# EOF
