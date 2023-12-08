# GMFU Optimizer Configurations
# Reconstructor: Star Liu <StarLiu@seas.upenn.edu>
# For license information, see LICENSE.TXT
"""Optimizer configuration"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from . import _read_yaml


@dataclass
class OptimizerConfig:
    """Optimizer and Learning Rate Scheduler configuration.
    Args:
        optimizer (str): Any of the standard optimizers from
                [torch.optim](https://pytorch.org/docs/stable/optim.html#algorithms).
        optimizer_params (Dict): The parameters for the optimizer. If left blank, will use default
                parameters.
        lr_scheduler (Optional[str]): The name of the LearningRateScheduler to use, if any, from
                [torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-
                rate). If None, will not use any scheduler. Defaults to `None`
        lr_scheduler_params (Optional[Dict]): The parameters for the LearningRateScheduler. If left blank,
                will use default parameters.
        lr_scheduler_monitor_metric (Optional[str]): Used with ReduceLROnPlateau, where the plateau is
                decided based on this metric
    """
    optimizer: str = field(
        default="Adam",
        metadata={
            "help": "Any of the standard optimizers from [torch.optim](https://pytorch.org/docs/stable/optim.html#algorithms)."
        },
    )
    optimizer_params: Dict = field(
        default_factory=lambda: {},
        metadata={"help": "The parameters for the optimizer. If left blank, will use default parameters."},
    )
    lr_scheduler: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the LearningRateScheduler to use, if any, from [torch.optim.lr_scheduler](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate). If None, will not use any scheduler. Defaults to `None`",
        },
    )
    lr_scheduler_params: Optional[Dict] = field(
        default_factory=lambda: {},
        metadata={"help": "The parameters for the LearningRateScheduler. If left blank, will use default parameters."},
    )
    lr_scheduler_monitor_metric: Optional[str] = field(
        default="valid_loss",
        metadata={"help": "Used with ReduceLROnPlateau, where the plateau is decided based on this metric"},
    )
    @staticmethod
    def read_from_yaml(filename: str = "config/optimizer_config.yml"):
        config = _read_yaml(filename)
        if config["lr_scheduler_params"] is None:
            config["lr_scheduler_params"] = {}
        return OptimizerConfig(**config)
