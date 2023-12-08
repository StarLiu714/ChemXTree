from dataclasses import dataclass, field
import math
from torch import nn
from ChemXTree.GMFU.utils import _initialize_layers, _linear_dropout_bn


@dataclass
class LinearHeadConfig:
    """A model class for Linear Head configuration; serves as a template and documentation. The models take a dictionary as input, but if there are keys which are not present in this model class, it'll throw an exception.
    Args:
        layers (str): Hyphen-separated number of layers and units in the classification/regression head.
                eg. 32-64-32. Default is just a mapping from intput dimension to output dimension

        activation (str): The activation type in the classification head. The default activaion in PyTorch
                like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-
                weighted-sum-nonlinearity

        dropout (float): probability of an classification element to be zeroed.

        use_batch_norm (bool): Flag to include a BatchNorm layer after each Linear Layer+DropOut

        initialization (str): Initialization scheme for the linear layers. Defaults to `kaiming`. Choices
                are: [`kaiming`,`xavier`,`random`].
    """

    layers: str = field(
        default="",
        metadata={
            "help": "Hyphen-separated number of layers and units in the classification/regression head. eg. 32-64-32. Default is just a mapping from intput dimension to output dimension"
        },
    )
    activation: str = field(
        default="ReLU",
        metadata={
            "help": "The activation type in the classification head. The default activaion in PyTorch like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "probability of an classification element to be zeroed."},
    )
    use_batch_norm: bool = field(
        default=False,
        metadata={"help": "Flag to include a BatchNorm layer after each Linear Layer+DropOut"},
    )
    initialization: str = field(
        default="kaiming",
        metadata={
            "help": "Initialization scheme for the linear layers. Defaults to `kaiming`",
            "choices": ["kaiming", "xavier", "random"],
        },
    )


def config_link(r):
    """
    This is a helper function decorator to link the config to the head.
    """

    def wrapper(f):
        f.config_template = r
        return f

    return wrapper


class Head(nn.Module):
    def __init__(self, layers, config_template, **kwargs):
        super().__init__()
        self.layers = layers
        self._config_template = config_template

    def forward(self, x):
        return self.layers(x)


class LinearHead(Head):
    _config_template = LinearHeadConfig

    def __init__(self, in_units: int, output_dim: int, config, **kwargs):
        # Linear Layers
        _layers = []
        _curr_units = in_units
        for units in config.layers.split("-"):
            try:
                int(units)
            except ValueError:
                if units == "":
                    continue
                else:
                    raise ValueError(f"Invalid units {units} in layers {config.layers}")
            _layers.extend(
                _linear_dropout_bn(
                    config.activation,
                    config.initialization,
                    config.use_batch_norm,
                    _curr_units,
                    int(units),
                    config.dropout,
                )
            )
            _curr_units = int(units)
        # Appending Final Output
        _layers.append(nn.Linear(_curr_units, output_dim))
        linear_layers = nn.Sequential(*_layers)
        _initialize_layers(config.activation, config.initialization, linear_layers)
        super().__init__(
            layers=linear_layers,
            config_template=LinearHeadConfig,
        )


ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
LOG2PI = math.log(2 * math.pi)