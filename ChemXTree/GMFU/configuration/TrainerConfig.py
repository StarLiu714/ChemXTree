# GMFU Trainer Configuration
# Reconstructor: Star <Star@seas.hahaha.edu>
# For license information, see LICENSE.TXT
""" Trainer configuration """
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from . import _validate_choices


@dataclass
class TrainerConfig:
    """Trainer configuration
    Args:
        batch_size (int): Number of samples in each batch of training
        data_aware_init_batch_size (int): Number of samples in each batch of training for the data-aware initialization, when applicable. Defaults to 2000
        fast_dev_run (bool): runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) of train, val
                and test to find any bugs (ie: a sort of unit test).
        max_epochs (int): Maximum number of epochs to be run
        min_epochs (Optional[int]): Force training for at least these many epochs. 1 by default
        max_time (Optional[int]): Stop training after this amount of time has passed. Disabled by default
                (None)
        gpus (Optional[int]): DEPRECATED: Number of gpus to train on (int). -1 uses all available GPUs. By
                default uses CPU (None)
        accelerator (Optional[str]): The accelerator to use for training. Can be one of
                'cpu','gpu','tpu','ipu','auto'. Defaults to 'auto'. Choices are: [`cpu`,`gpu`,`tpu`,`ipu`,`auto`].
        devices (Optional[int]): Number of devices to train on (int). -1 uses all available devices. By
                default uses all available devices (-1)
        devices_list (Optional[List[int]]): List of devices to train on (list). If specified, takes
                precedence over `devices` argument. Defaults to None
        accumulate_grad_batches (int): Accumulates grads every k batches or as set up in the dict. Trainer
                also calls optimizer.step() for the last indivisible step number.
        auto_lr_find (bool): Runs a learning rate finder algorithm (see this paper) when calling
                trainer.tune(), to find optimal initial learning rate.
        auto_select_gpus (bool): If enabled and `devices` is an integer, pick available gpus automatically.
                This is especially useful when GPUs are configured to be in 'exclusive mode', such that only one
                process at a time can access them.
        check_val_every_n_epoch (int): Check val every n train epochs.
        gradient_clip_val (float): Gradient clipping value
        overfit_batches (float): Uses this much data of the training set. If nonzero, will use the same
                training set for validation and testing. If the training dataloaders have shuffle=True, Lightning
                will automatically disable it. Useful for quickly debugging or trying to overfit on purpose.
        deterministic (bool): If true enables cudnn.deterministic. Might make your system slower, but
                ensures reproducibility.
        profiler (Optional[str]): To profile individual steps during training and assist in identifying
                bottlenecks. None, simple or advanced, pytorch. Choices are:
                [`None`,`simple`,`advanced`,`pytorch`].
        early_stopping (Optional[str]): The loss/metric that needed to be monitored for early stopping. If
                None, there will be no early stopping
        early_stopping_min_delta (float): The minimum delta in the loss/metric which qualifies as an
                improvement in early stopping
        early_stopping_mode (str): The direction in which the loss/metric should be optimized. Choices are:
                [`max`,`min`].
        early_stopping_patience (int): The number of epochs to wait until there is no further improvements
                in loss/metric
        early_stopping_kwargs (Optional[Dict]): Additional keyword arguments for the early stopping callback.
                See the documentation for the PyTorch Lightning EarlyStopping callback for more details.
        checkpoints (Optional[str]): The loss/metric that needed to be monitored for checkpoints. If None,
                there will be no checkpoints
        checkpoints_path (str): The path where the saved models will be
        checkpoints_every_n_epochs (int): Number of training steps between checkpoints
        checkpoints_name (Optional[str]): The name under which the models will be saved. If left blank,
                first it will look for `run_name` in experiment_config and if that is also None then it will use a
                generic name like task_version.
        checkpoints_mode (str): The direction in which the loss/metric should be optimized
        checkpoints_save_top_k (int): The number of best models to save
        checkpoints_kwargs (Optional[Dict]): Additional keyword arguments for the checkpoints callback.
                See the documentation for the PyTorch Lightning ModelCheckpoint callback for more details.
        load_best (bool): Flag to load the best model saved during training
        track_grad_norm (int): Track and Log Gradient Norms in the logger. -1 by default means no tracking.
                1 for the L1 norm, 2 for L2 norm, etc.
        progress_bar (str): Progress bar type. Can be one of: `none`, `simple`, `rich`. Defaults to `rich`.
        precision (int): Precision of the model. Can be one of: `32`, `16`, `64`. Defaults to `32`..
                Choices are: [`32`,`16`,`64`].
        seed (int): Seed for random number generators. Defaults to 42
        trainer_kwargs (Dict[str, Any]): Additional kwargs to be passed to PyTorch Lightning Trainer. See
                https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.html#pytorch_lightning.trainer.Trainer
    """
    batch_size: int = field(default=64, metadata={"help": "Number of samples in each batch of training"})
    data_aware_init_batch_size: int = field(
        default=2000,
        metadata={
            "help": "Number of samples in each batch of training for the data-aware initialization, when applicable. Defaults to 2000"
        },
    )
    fast_dev_run: bool = field(
        default=False,
        metadata={
            "help": "runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es) of train, val and test to find any bugs (ie: a sort of unit test)."
        },
    )
    max_epochs: int = field(default=10, metadata={"help": "Maximum number of epochs to be run"})
    min_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "Force training for at least these many epochs. 1 by default"},
    )
    max_time: Optional[int] = field(
        default=None,
        metadata={"help": "Stop training after this amount of time has passed. Disabled by default (None)"},
    )
    gpus: Optional[int] = field(
        default=None,
        metadata={
            "help": "DEPRECATED: Number of gpus to train on (int). -1 uses all available GPUs. By default uses CPU (None)"
        },
    )
    accelerator: Optional[str] = field(
        default="auto",
        metadata={
            "help": "The accelerator to use for training. Can be one of 'cpu','gpu','tpu','ipu','auto'. Defaults to 'auto'",
            "choices": ["cpu", "gpu", "tpu", "ipu", "auto"],
        },
    )
    devices: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of devices to train on (int). -1 uses all available devices. By default uses all available devices (-1)",
        },
    )
    devices_list: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "List of devices to train on (list). If specified, takes precedence over `devices` argument. Defaults to None",
        },
    )
    accumulate_grad_batches: int = field(
        default=1,
        metadata={
            "help": "Accumulates grads every k batches or as set up in the dict. Trainer also calls optimizer.step() for the last indivisible step number."
        },
    )
    auto_lr_find: bool = field(
        default=False,
        metadata={
            "help": "Runs a learning rate finder algorithm (see this paper) when calling trainer.tune(), to find optimal initial learning rate."
        },
    )
    auto_select_gpus: bool = field(
        default=True,
        metadata={
            "help": "If enabled and `devices` is an integer, pick available gpus automatically. This is especially useful when GPUs are configured to be in 'exclusive mode', such that only one process at a time can access them."
        },
    )
    check_val_every_n_epoch: int = field(default=1, metadata={"help": "Check val every n train epochs."})
    gradient_clip_val: float = field(default=0.0, metadata={"help": "Gradient clipping value"})
    overfit_batches: float = field(
        default=0.0,
        metadata={
            "help": "Uses this much data of the training set. If nonzero, will use the same training set for validation and testing. If the training dataloaders have shuffle=True, Lightning will automatically disable it. Useful for quickly debugging or trying to overfit on purpose."
        },
    )
    deterministic: bool = field(
        default=False,
        metadata={
            "help": "If true enables cudnn.deterministic. Might make your system slower, but ensures reproducibility."
        },
    )
    profiler: Optional[str] = field(
        default=None,
        metadata={
            "help": "To profile individual steps during training and assist in identifying bottlenecks. None, simple or advanced, pytorch",
            "choices": [None, "simple", "advanced", "pytorch"],
        },
    )
    early_stopping: Optional[str] = field(
        default="valid_loss",
        metadata={
            "help": "The loss/metric that needed to be monitored for early stopping. If None, there will be no early stopping"
        },
    )
    early_stopping_min_delta: float = field(
        default=0.001,
        metadata={"help": "The minimum delta in the loss/metric which qualifies as an improvement in early stopping"},
    )
    early_stopping_mode: str = field(
        default="min",
        metadata={
            "help": "The direction in which the loss/metric should be optimized",
            "choices": ["max", "min"],
        },
    )
    early_stopping_patience: int = field(
        default=3,
        metadata={"help": "The number of epochs to wait until there is no further improvements in loss/metric"},
    )
    early_stopping_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: dict(),
        metadata={
            "help": "Additional keyword arguments for the early stopping callback. See the documentation for the PyTorch Lightning EarlyStopping callback for more details."
        },
    )
    checkpoints: Optional[str] = field(
        default="valid_loss",
        metadata={
            "help": "The loss/metric that needed to be monitored for checkpoints. If None, there will be no checkpoints"
        },
    )
    checkpoints_path: str = field(
        default="saved_models",
        metadata={"help": "The path where the saved models will be"},
    )
    checkpoints_every_n_epochs: int = field(
        default=1,
        metadata={"help": "Number of training steps between checkpoints"},
    )
    checkpoints_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name under which the models will be saved. If left blank, first it will look for `run_name` in experiment_config and if that is also None then it will use a generic name like task_version."
        },
    )
    checkpoints_mode: str = field(
        default="min",
        metadata={"help": "The direction in which the loss/metric should be optimized"},
    )
    checkpoints_save_top_k: int = field(
        default=1,
        metadata={"help": "The number of best models to save"},
    )
    checkpoints_kwargs: Optional[Dict[str, Any]] = field(
        default_factory=lambda: dict(),
        metadata={
            "help": "Additional keyword arguments for the checkpoints callback. See the documentation for the PyTorch Lightning ModelCheckpoint callback for more details."
        },
    )
    load_best: bool = field(
        default=True,
        metadata={"help": "Flag to load the best model saved during training"},
    )
    track_grad_norm: int = field(
        default=-1,
        metadata={
            "help": "Track and Log Gradient Norms in the logger. -1 by default means no tracking. 1 for the L1 norm, 2 for L2 norm, etc."
        },
    )
    progress_bar: str = field(
        default="rich",
        metadata={"help": "Progress bar type. Can be one of: `none`, `simple`, `rich`. Defaults to `rich`."},
    )
    precision: int = field(
        default=32,
        metadata={
            "help": "Precision of the model. Can be one of: `32`, `16`, `64`. Defaults to `32`.",
            "choices": [32, 16, 64],
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "Seed for random number generators. Defaults to 42"},
    )
    trainer_kwargs: Dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Additional kwargs to be passed to PyTorch Lightning Trainer. See https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.html#pytorch_lightning.trainer.Trainer"
        },
    )
    def __post_init__(self):
        _validate_choices(self)
        if self.gpus is not None:
            warnings.warn(
                "The `gpus` argument is deprecated in favor of `accelerator` and will be removed in a future version. Please use `accelerator='gpu'` instead.",
                DeprecationWarning,
            )
            if self.devices is None:
                self.devices = self.gpus
            if self.accelerator is None:
                self.accelerator = "gpu"
        else:
            if self.accelerator is None:
                self.accelerator = "cpu"
        delattr(self, "gpus")
        if self.devices_list is not None:
            warnings.warn("Ignoring devices in favor of devices_list")
            self.devices = self.devices_list
        delattr(self, "devices_list")
        for key in self.early_stopping_kwargs.keys():
            if key in ["min_delta", "mode", "patience"]:
                raise ValueError(
                    f"Cannot override {key} in early_stopping_kwargs. Please use the appropriate argument in `TrainerConfig`"
                )
        for key in self.checkpoints_kwargs.keys():
            if key in ["dirpath", "filename", "monitor", "save_top_k", "mode", "every_n_epochs"]:
                raise ValueError(
                    f"Cannot override {key} in checkpoints_kwargs. Please use the appropriate argument in `TrainerConfig`"
                )