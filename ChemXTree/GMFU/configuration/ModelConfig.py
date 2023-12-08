# GMFU Model Configurations
# Reconstructor: Star Liu <StarLiu@seas.upenn.edu>
# For license information, see LICENSE.TXT
"""Model configuration"""
import re
from dataclasses import dataclass, field
from typing import Dict,List, Optional
from ChemXTree.GMFU.model.blocks import heads as heads_dir
from ChemXTree.GMFU.utils import get_logger
from . import _validate_choices


logger = get_logger(__name__)


@dataclass
class ModelConfig:
    """Base Model configuration
    Args:
        task (str): Specify whether the problem is regression or classification. `backbone` is a task which
                considers the model as a backbone to generate features. Choices are: [`regression`,`classification`,`backbone`].
        head (Optional[str]): The head to be used for the model. Should be one of the heads defined in
                `ChemXTree.GMFU.model.blocks.heads`. Defaults to  LinearHead. Choices are:
                [`None`,`LinearHead`,`MixtureDensityHead`].
        head_config (Optional[Dict]): The config as a dict which defines the head. If left empty, will be
                initialized as default linear head.
        embedding_dims (Optional[List]): The dimensions of the embedding for each categorical column as a
                list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of
                the categorical column using the rule min(50, (x + 1) // 2)
        embedding_dropout (float): Dropout to be applied to the Categorical Embedding. Defaults to 0.0
        batch_norm_continuous_input (bool): If True, we will normalize the continuous layer by passing it
                through a BatchNorm layer.
        learning_rate (float): The learning rate of the model. Defaults to 1e-3.
        loss (Optional[str]): The loss function to be applied. By Default it is MSELoss for regression and
                CrossEntropyLoss for classification. Unless you are sure what you are doing, leave it at MSELoss
                or L1Loss for regression and CrossEntropyLoss for classification
        metrics (Optional[List[str]]): the list of metrics you need to track during training. The metrics
                should be one of the functional metrics implemented in ``torchmetrics``. By default, it is
                accuracy if classification and mean_squared_error for regression
        metrics_params (Optional[List]): The parameters to be passed to the metrics function
        target_range (Optional[List]): The range in which we should limit the output variable. Currently
                ignored for multi-target regression. Typically used for Regression problems. If left empty, will
                not apply any restrictions
        seed (int): The seed for reproducibility. Defaults to 42
    """
    task: str = field(
        metadata={
            "help": "Specify whether the problem is regression or classification. `backbone` is a task which considers the model as a backbone to generate features.",
            "choices": ["regression", "classification", "backbone"],
        }
    )
    head: Optional[str] = field(
        default="LinearHead",
        metadata={
            "help": "The head to be used for the model. Should be one of the heads defined in `ChemXTree.GMFU.model.blocks.heads`. Defaults to  LinearHead",
            "choices": [None, "LinearHead", "MixtureDensityHead"],
        },
    )
    head_config: Optional[Dict] = field(
        default_factory=lambda: {"layers": ""},
        metadata={
            "help": "The config as a dict which defines the head. If left empty, will be initialized as default linear head."
        },
    )
    embedding_dims: Optional[List] = field(
        default=None,
        metadata={
            "help": "The dimensions of the embedding for each categorical column as a list of tuples "
            "(cardinality, embedding_dim). If left empty, will infer using the cardinality of the "
            "categorical column using the rule min(50, (x + 1) // 2)"
        },
    )
    embedding_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout to be applied to the Categorical Embedding. Defaults to 0.0"},
    )
    batch_norm_continuous_input: bool = field(
        default=True,
        metadata={"help": "If True, we will normalize the continuous layer by passing it through a BatchNorm layer."},
    )
    learning_rate: float = field(
        default=1e-3,
        metadata={"help": "The learning rate of the model. Defaults to 1e-3."},
    )
    loss: Optional[str] = field(
        default=None,
        metadata={
            "help": "The loss function to be applied. By Default it is MSELoss for regression "
            "and CrossEntropyLoss for classification. Unless you are sure what you are doing, "
            "leave it at MSELoss or L1Loss for regression and CrossEntropyLoss for classification"
        },
    )
    metrics: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "the list of metrics you need to track during training. The metrics should be one "
            "of the functional metrics implemented in ``torchmetrics``. By default, "
            "it is accuracy if classification and mean_squared_error for regression"
        },
    )
    metrics_params: Optional[List] = field(
        default=None,
        metadata={"help": "The parameters to be passed to the metrics function"},
    )
    target_range: Optional[List] = field(
        default=None,
        metadata={
            "help": "The range in which we should limit the output variable. "
            "Currently ignored for multi-target regression. Typically used for Regression problems. "
            "If left empty, will not apply any restrictions"
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "The seed for reproducibility. Defaults to 42"},
    )
    _module_src: str = field(default="model")
    _model_name: str = field(default="Model")
    _backbone_name: str = field(default="Backbone")
    _config_name: str = field(default="Config")

    def __post_init__(self):
        if self.task == "regression":
            self.loss = "MSELoss" if self.loss is None else self.loss
            self.metrics = ["mean_squared_error"] if self.metrics is None else self.metrics
            self.metrics_params = [{} for _ in self.metrics] if self.metrics_params is None else self.metrics_params
        elif self.task == "classification":
            self.loss = "CrossEntropyLoss" if self.loss is None else self.loss
            self.metrics = ["accuracy"] if self.metrics is None else self.metrics
            self.metrics_params = [{} for _ in self.metrics] if self.metrics_params is None else self.metrics_params
        elif self.task == "backbone":
            self.loss = None
            self.metrics = None
            self.metrics_params = None
            if self.head is not None:
                logger.warning("`head` is not a valid parameter for backbone task. Making `head=None`")
                self.head = None
                self.head_config = None
        else:
            raise NotImplementedError(
                f"{self.task} is not a valid task. Should be one of "
                f"{self.__dataclass_fields__['task'].metadata['choices']}"
            )
        if self.metrics is not None:
            assert len(self.metrics) == len(self.metrics_params), "metrics and metric_params should have same length"
        if self.task != "backbone":
            assert self.head in dir(heads_dir), f"{self.head} is not a valid head"
            _head_callable = getattr(heads_dir, self.head)
            ideal_head_config = _head_callable._config_template
            invalid_keys = set(self.head_config.keys()) - set(ideal_head_config.__dict__.keys())
            assert len(invalid_keys) == 0, f"`head_config` has some invalid keys: {invalid_keys}"
        # For Custom models, setting these values for compatibility
        if not hasattr(self, "_config_name"):
            self._config_name = type(self).__name__
        if not hasattr(self, "_model_name"):
            self._model_name = re.sub("[Cc]onfig", "Model", self._config_name)
        if not hasattr(self, "_backbone_name"):
            self._backbone_name = re.sub("[Cc]onfig", "Backbone", self._config_name)
        _validate_choices(self)
