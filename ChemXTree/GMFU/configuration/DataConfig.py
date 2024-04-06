# GMFU data configuration
# Reconstructor: Star Liu <StarLiu@seas.upenn.edu>
# For license information, see LICENSE.TXT
""" Data configuration"""
import os
from typing import List, Optional
from dataclasses import dataclass, field
from . import _validate_choices

@dataclass
class DataConfig:
    """Data configuration.
    Args:
        target (Optional[List[str]]): A list of strings with the names of the target column(s). It is
                mandatory for all.
        continuous_cols (List): Column names of the numeric fields. Defaults to []
        categorical_cols (List): Column names of the categorical fields to treat differently. Defaults to
                []
        date_columns (List): (Column names, Freq) tuples of the date fields. For eg. a field named
                introduction_date and with a monthly frequency should have an entry ('intro_date','M'}
        encode_date_columns (bool): Whether or not to encode the derived variables from date
        validation_split (Optional[float]): Percentage of Training rows to keep aside as validation. Used
                only if Validation Data is not given separately
        continuous_feature_transform (Optional[str]): Whether or not to transform the features before
                modelling. By default it is turned off.. Choices are: [`None`,`yeo-johnson`,`box-
                cox`,`quantile_normal`,`quantile_uniform`].
        normalize_continuous_features (bool): Flag to normalize the input features(continuous)
        quantile_noise (int): NOT IMPLEMENTED. If specified fits QuantileTransformer on data with added
                gaussian noise with std = :quantile_noise: * data.std ; this will cause discrete values to be more
                separable. Please not that this transformation does NOT apply gaussian noise to the resulting
                data, the noise is only applied for QuantileTransformer
        num_workers (Optional[int]): The number of workers used for data loading. For windows always set to
                0
        pin_memory (bool): Whether or not to pin memory for data loading.
        handle_unknown_categories (bool): Whether or not to handle unknown or new values in categorical
                columns as unknown
        handle_missing_values (bool): Whether or not to handle missing values in categorical columns as
                unknown
    """
    target: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "A list of strings with the names of the target column(s). It is mandatory for all."
        },
    )
    continuous_cols: List = field(
        default_factory=list,
        metadata={"help": "Column names of the numeric fields. Defaults to []"},
    )
    categorical_cols: List = field(
        default_factory=list,
        metadata={"help": "Column names of the categorical fields to treat differently. Defaults to []"},
    )
    date_columns: List = field(
        default_factory=list,
        metadata={
            "help": "(Column names, Freq) tuples of the date fields. For eg. a field named introduction_date and with a monthly frequency should have an entry ('intro_date','M'}"
        },
    )
    encode_date_columns: bool = field(
        default=True,
        metadata={"help": "Whether or not to encode the derived variables from date"},
    )
    validation_split: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "Percentage of Training rows to keep aside as validation. Used only if Validation Data is not given separately"
        },
    )
    continuous_feature_transform: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether or not to transform the features before modelling. By default it is turned off.",
            "choices": [
                None,
                "yeo-johnson",
                "box-cox",
                "quantile_normal",
                "quantile_uniform",
            ],
        },
    )
    normalize_continuous_features: bool = field(
        default=True,
        metadata={"help": "Flag to normalize the input features(continuous)"},
    )
    quantile_noise: int = field(
        default=0,
        metadata={
            "help": "NOT IMPLEMENTED. If specified fits QuantileTransformer on data with added gaussian noise with std = :quantile_noise: * data.std ; this will cause discrete values to be more separable. Please not that this transformation does NOT apply gaussian noise to the resulting data, the noise is only applied for QuantileTransformer"
        },
    )
    num_workers: Optional[int] = field(
        default=0,
        metadata={"help": "The number of workers used for data loading. For windows always set to 0"},
    )
    pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether or not to pin memory for data loading."},
    )
    handle_unknown_categories: bool = field(
        default=True,
        metadata={"help": "Whether or not to handle unknown or new values in categorical columns as unknown"},
    )
    handle_missing_values: bool = field(
        default=True,
        metadata={"help": "Whether or not to handle missing values in categorical columns as unknown"},
    )
    def __post_init__(self):
        assert (
            len(self.categorical_cols) + len(self.continuous_cols) + len(self.date_columns) > 0
        ), "There should be at-least one feature defined in categorical, continuous, or date columns"
        _validate_choices(self)
        if os.name == "nt" and self.num_workers != 0:
            print("Windows does not support num_workers > 0. Setting num_workers to 0")
            self.num_workers = 0