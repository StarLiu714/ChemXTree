# GMFU Utility Functions
# Reconstructor: Star Liu <StarLiu@seas.upenn.edu>
# For license information, see LICENSE.TXT
"""Utility functions"""
import logging
import os
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, IO, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning_lite.utilities.cloud_io import get_filesystem
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

import ChemXTree.GMFU as root_module

_PATH = Union[str, Path]
_DEVICE = Union[torch.device, str, int]
_MAP_LOCATION_TYPE = Optional[Union[_DEVICE, Callable[[_DEVICE], _DEVICE], Dict[_DEVICE, _DEVICE]]]


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(level=os.environ.get("PT_LOGLEVEL", "INFO"))
    formatter = logging.Formatter("%(asctime)s - {%(name)s:%(lineno)d} - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = logging.getLogger(__name__)


def _make_smooth_weights_for_balanced_classes(y_train, mu=1.0):
    labels_dict = {label: count for label, count in zip(np.unique(y_train), np.bincount(y_train))}
    total = np.sum(list(labels_dict.values()))
    keys = sorted(labels_dict.keys())
    weight = []
    for i in keys:
        score = np.log(mu * total / float(labels_dict[i]))
        weight.append(score if score > 1 else 1)
    return weight


def get_class_weighted_cross_entropy(y_train, mu=1.0):
    y_train = LabelEncoder().fit_transform(y_train)
    weights = _make_smooth_weights_for_balanced_classes(y_train, mu=mu)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights))
    return criterion
def get_multiclass_weighted_cross_entropy(y_train, mu=1.0):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterions = []
    for i in range(y_train.shape[1]):
        y_encoded = LabelEncoder().fit_transform(y_train[:, i])
        weights = _make_smooth_weights_for_balanced_classes(y_encoded, mu=mu)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))
        criterions.append(criterion)
    return criterions


def get_multitask_loss(y1_train, y2_train, mu=1.0):
    assert y1_train.ndim == 1 and y2_train.ndim == 1, "Utility function only works for binary classification"
    y1_train = LabelEncoder().fit_transform(y1_train)
    y2_train = LabelEncoder().fit_transform(y2_train)
    weights1 = _make_smooth_weights_for_balanced_classes(y1_train, mu=mu)
    weights2 = _make_smooth_weights_for_balanced_classes(y2_train, mu=mu)
    criterion1 = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights1))
    criterion2 = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights2))
    def loss_func(y1_pred, y2_pred, y1_true, y2_true):
        return criterion1(y1_pred, y1_true) + criterion2(y2_pred, y2_true)
    return loss_func


def _initialize_layers(activation, initialization, layers):
    if type(layers) == nn.Sequential:
        for layer in layers:
            if hasattr(layer, "weight"):
                _initialize_layers(activation, initialization, layer)
    else:
        if activation == "ReLU":
            nonlinearity = "relu"
        elif activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            if initialization == "kaiming":
                logger.warning("Kaiming initialization is only recommended for ReLU and LeakyReLU.")
                nonlinearity = "leaky_relu"
            else:
                nonlinearity = "relu"

        if initialization == "kaiming":
            nn.init.kaiming_normal_(layers.weight, nonlinearity=nonlinearity)
        elif initialization == "xavier":
            nn.init.xavier_normal_(
                layers.weight,
                gain=nn.init.calculate_gain(nonlinearity) if activation in ["ReLU", "LeakyReLU"] else 1,
            )
        elif initialization == "random":
            nn.init.normal_(layers.weight)


def _linear_dropout_bn(activation, initialization, use_batch_norm, in_units, out_units, dropout):
    if isinstance(activation, str):
        _activation = getattr(nn, activation)
    else:
        _activation = activation
    layers = []
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(num_features=in_units))
    linear = nn.Linear(in_units, out_units)
    _initialize_layers(activation, initialization, linear)
    layers.extend([linear, _activation()])
    if dropout != 0:
        layers.append(nn.Dropout(dropout))
    return layers


def getattr_nested(_module_src, _model_name):
    module = root_module
    for m in _module_src.split("."):
        module = getattr(module, m)
    return getattr(module, _model_name)



def generate_doc_dataclass(dataclass, desc=None, width=100):
    if desc is not None:
        doc_str = f"{desc}\nArgs:"
    else:
        doc_str = "Args:"
    for key in dataclass.__dataclass_fields__.keys():
        if key.startswith("_"):  # Skipping private fields
            continue
        atr = dataclass.__dataclass_fields__[key]
        if atr.init:
            type = str(atr.type).replace("<class '", "").replace("'>", "").replace("typing.", "")
            help_str = atr.metadata.get("help", "")
            if "choices" in atr.metadata.keys():
                help_str += f'. Choices are: [{",".join(["`"+str(ch)+"`" for ch in atr.metadata["choices"]])}].'
            # help_str += f'. Defaults to {atr.default}'
            h_str = textwrap.fill(
                f"{key} ({type}): {help_str}",
                width=width,
                subsequent_indent="\t\t",
                initial_indent="\t",
            )
            h_str = f"\n{h_str}\n"
            doc_str += h_str
    return doc_str


# Copied over pytorch_lightning.utilities.cloud_io.load as it was deprecated
def pl_load(
    path_or_url: Union[IO, _PATH],
    map_location: _MAP_LOCATION_TYPE = None,
) -> Any:
    """Loads a checkpoint.

    Args:
        path_or_url: Path or URL of the checkpoint.
        map_location: a function, ``torch.device``, string or a dict specifying how to remap storage locations.
    """
    if not isinstance(path_or_url, (str, Path)):
        # any sort of BytesIO or similar
        return torch.load(path_or_url, map_location=map_location)
    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(
            str(path_or_url),
            map_location=map_location,  # type: ignore[arg-type] # upstream annotation is not correct
        )
    fs = get_filesystem(path_or_url)
    with fs.open(path_or_url, "rb") as f:
        return torch.load(f, map_location=map_location)


def reset_all_weights(model: nn.Module) -> None:
    """
    Resets all parameters in a network.

    Args:
        model: The model to reset the parameters of.

    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)