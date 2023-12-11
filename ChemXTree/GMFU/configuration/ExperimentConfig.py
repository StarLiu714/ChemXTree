# GMFU Experiment Configurations
# Reconstructor: Star <Star@seas.hahaha.edu>
# For license information, see LICENSE.TXT
"""Experiment configuration"""

from dataclasses import dataclass, field, MISSING
from typing import Optional
from . import _validate_choices

@dataclass
class ExperimentConfig:
    """Experiment configuration. Experiment Tracking with WandB and Tensorboard
    Args:
        project_name (str): The name of the project under which all runs will be logged. For Tensorboard
                this defines the folder under which the logs will be saved and for W&B it defines the project name
        run_name (Optional[str]): The name of the run; a specific identifier to recognize the run. If left
                blank, will be assigned a auto-generated name
        exp_watch (Optional[str]): The level of logging required.  Can be `gradients`, `parameters`, `all`
                or `None`. Defaults to None. Choices are: [`gradients`,`parameters`,`all`,`None`].
        log_target (str): Determines where logging happens - Tensorboard or W&B. Choices are:
                [`wandb`,`tensorboard`].
        log_logits (bool): Turn this on to log the logits as a histogram in W&B
        exp_log_freq (int): step count between logging of gradients and parameters.
    """
    project_name: str = field(
        default=MISSING,
        metadata={
            "help": "The name of the project under which all runs will be logged. For Tensorboard this defines the folder under which the logs will be saved and for W&B it defines the project name"
        },
    )
    run_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the run; a specific identifier to recognize the run. If left blank, will be assigned a auto-generated name"
        },
    )
    exp_watch: Optional[str] = field(
        default=None,
        metadata={
            "help": "The level of logging required.  Can be `gradients`, `parameters`, `all` or `None`. Defaults to None",
            "choices": ["gradients", "parameters", "all", None],
        },
    )
    log_target: str = field(
        default="tensorboard",
        metadata={
            "help": "Determines where logging happens - Tensorboard or W&B",
            "choices": ["wandb", "tensorboard"],
        },
    )
    log_logits: bool = field(
        default=False,
        metadata={"help": "Turn this on to log the logits as a histogram in W&B"},
    )
    exp_log_freq: int = field(
        default=100,
        metadata={"help": "step count between logging of gradients and parameters."},
    )
    def __post_init__(self):
        _validate_choices(self)
        if self.log_target == "wandb":
            try:
                import wandb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "No W&B installation detected. `pip install wandb` to install W&B if you set log_target as `wandb`"
                )