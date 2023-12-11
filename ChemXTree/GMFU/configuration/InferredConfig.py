# GMFU Inferred Configurations
# Reconstructor: Star <Star@seas.hahaha.edu>
# For license information, see LICENSE.TXT
""" Inferred configuration"""
from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass
class InferredConfig:
    """
    Configuration inferred from the data during `fit` of the TabularDatamodule
    Args:
        num_tasks (int): Number of tasks. When >1 is multi-class.
        n_heads (int): Number of heads for multi-head attention.
        residual_scale (float): Scale of residual layer. .
        categorical_dim (int): The number of categorical features
        continuous_dim (int): The number of continuous features
        output_dim (Optional[int]): The number of output targets
        categorical_cardinality (Optional[List[int]]): The number of unique values in categorical features
        embedding_dims (Optional[List]): The dimensions of the embedding for each categorical column as a
                list of tuples (cardinality, embedding_dim).
        embedded_cat_dim (int): The number of features or dimensions of the embedded categorical features
    """
    num_tasks: int = field(
        metadata={"help": "The number of tasks"},
    )
    n_heads: int = field(
        metadata={"help": "The number of heads for multi-head attention"},
    )
    residual_scale: Optional[float] = field(
        metadata={"help": "Scale of residual layer. "}
    )
    categorical_dim: int = field(
        metadata={"help": "The number of categorical features"},
    )
    continuous_dim: int = field(
        metadata={"help": "The number of continuous features"},
    )
    output_dim: Optional[int] = field(
        default=None,
        metadata={"help": "The number of output targets"},
    )
    categorical_cardinality: Optional[List[int]] = field(
        default=None,
        metadata={"help": "The number of unique values in categorical features"},
    )
    embedding_dims: Optional[List] = field(
        default=None,
        metadata={
            "help": "The dimensions of the embedding for each categorical column as a list of tuples "
            "(cardinality, embedding_dim)."
        },
    )
    embedded_cat_dim: int = field(
        init=False,
        metadata={"help": "The number of features or dimensions of the embedded categorical features"},
    )
    def __post_init__(self):
        if self.embedding_dims is not None:
            assert all(
                [(isinstance(t, Iterable) and len(t) == 2) for t in self.embedding_dims]
            ), "embedding_dims must be a list of tuples (cardinality, embedding_dim)"
            self.embedded_cat_dim = sum([t[1] for t in self.embedding_dims])
        else:
            self.embedded_cat_dim = 0