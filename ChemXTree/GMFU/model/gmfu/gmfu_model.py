# GMFU model
# Reconstructor: Star Liu <StarLiu@seas.upenn.edu>
# For license information, see LICENSE.TXT
import torch
import torch.nn as nn
from omegaconf import DictConfig

from ChemXTree.GMFU.model.blocks.activatiors import entmax15, entmoid15, sparsemax, sparsemoid
from ChemXTree.GMFU.model.blocks import heads
from ChemXTree.GMFU.model.blocks.layers import Embedding1dLayer
from ChemXTree.GMFU.utils import get_logger

from ..framework import TablFramework
from .components import GFLUnit, NeuralDecisionTree

logger = get_logger(__name__)


class GateModulationFeatureUnitBackbone(nn.Module):
    ACTIVATION_MAP = {
        "entmax": entmax15,
        "sparsemax": sparsemax,
        "softmax": nn.functional.softmax,
    }

    BINARY_ACTIVATION_MAP = {
        "entmoid": entmoid15,
        "sparsemoid": sparsemoid,
        "sigmoid": nn.functional.sigmoid,
    }

    def __init__(
        self,
        cat_embedding_dims: list,
        n_continuous_features: int,
        gflu_stages: int,
        num_trees: int,
        tree_depth: int,
        num_tasks :int,
        n_heads: int,
        residual_scale: float = 0.05,
        chain_trees: bool = True,
        tree_wise_attention: bool = False,
        tree_wise_attention_dropout: float = 0.0,
        gflu_dropout: float = 0.0,
        tree_dropout: float = 0.0,
        binning_activation: str = "entmoid",
        feature_mask_function: str = "softmax",
        batch_norm_continuous_input: bool = True,
        embedding_dropout: float = 0.0,
        
    ):
        super().__init__()
        assert (
            binning_activation in self.BINARY_ACTIVATION_MAP.keys()
        ), f"`binning_activation should be one of {self.BINARY_ACTIVATION_MAP.keys()}"
        assert (
            feature_mask_function in self.ACTIVATION_MAP.keys()
        ), f"`feature_mask_function should be one of {self.ACTIVATION_MAP.keys()}"

        self.gflu_stages = gflu_stages
        self.num_trees = num_trees
        self.num_tasks = num_tasks
        self.n_heads = n_heads
        self.residual_scale = residual_scale
        self.tree_depth = tree_depth
        self.chain_trees = chain_trees
        self.tree_wise_attention = tree_wise_attention
        self.tree_wise_attention_dropout = tree_wise_attention_dropout
        self.gflu_dropout = gflu_dropout
        self.tree_dropout = tree_dropout
        self.binning_activation = self.BINARY_ACTIVATION_MAP[binning_activation]
        self.feature_mask_function = self.ACTIVATION_MAP[feature_mask_function]
        self.batch_norm_continuous_input = batch_norm_continuous_input
        self.n_continuous_features = n_continuous_features
        self.cat_embedding_dims = cat_embedding_dims
        self._embedded_cat_features = sum([y for x, y in cat_embedding_dims])
        self.n_features = self._embedded_cat_features + n_continuous_features
        self.embedding_dropout = embedding_dropout
        self.output_dim = 2**self.tree_depth
        self._build_network()

    def _build_network(self):
        if self.gflu_stages > 0:
            self.gflus = GFLUnit(
                n_features_in=self.n_features,
                n_stages=self.gflu_stages,
                feature_mask_function=self.feature_mask_function,
                dropout=self.gflu_dropout,
                n_heads=self.n_heads,
                residual_scale = self.residual_scale
            )
        self.trees = nn.ModuleList(
            [
                NeuralDecisionTree(
                    depth=self.tree_depth,
                    n_features=self.n_features + 2**self.tree_depth * t if self.chain_trees else self.n_features,
                    dropout=self.tree_dropout,
                    binning_activation=self.binning_activation,
                    feature_mask_function=self.feature_mask_function,
                )
                for t in range(self.num_trees)
            ]
        )
        if self.tree_wise_attention:
            self.tree_attention = nn.MultiheadAttention(
                self.output_dim,
                1,
                dropout=self.tree_wise_attention_dropout,
            )

    def _build_embedding_layer(self):
        return Embedding1dLayer(
            continuous_dim=self.n_continuous_features,
            categorical_embedding_dims=self.cat_embedding_dims,
            embedding_dropout=self.embedding_dropout,
            batch_norm_continuous_input=self.batch_norm_continuous_input,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gflu_stages > 0:
            x = self.gflus(x)
        # Decision Tree
        tree_outputs = []
        tree_feature_masks = []
        tree_input = x
        for i in range(self.num_trees):
            tree_output, feat_masks = self.trees[i](tree_input)
            tree_outputs.append(tree_output.unsqueeze(-1))
            tree_feature_masks.append(feat_masks)
            if self.chain_trees:
                tree_input = torch.cat([tree_input, tree_output], 1)
        tree_outputs = torch.cat(tree_outputs, dim=-1)
        if self.tree_wise_attention:
            tree_outputs, _ = self.tree_attention(tree_outputs)
        return tree_outputs


class CustomHead(nn.Module):
    """Custom Head for GMFU

    Args:
        input_dim (int): Input dimension of the head
        hparams (DictConfig): Config of the model
    """

    def __init__(self, input_dim: int, hparams: DictConfig):
        super().__init__()
        self.hparams = hparams
        self.input_dim = input_dim
        if self.hparams.share_head_weights:
            self.head = self._get_head_from_config()
        else:
            self.head = nn.ModuleList([self._get_head_from_config() for _ in range(self.hparams.num_trees)])
        # random parameter with num_trees elements
        self.eta = nn.Parameter(torch.rand(self.hparams.num_trees, requires_grad=True))
        if self.hparams.task == "regression":
            self.T0 = nn.Parameter(torch.rand(self.hparams.output_dim), requires_grad=True)

    def _get_head_from_config(self):
        _head_callable = getattr(heads, self.hparams.head)
        return _head_callable(
            in_units=self.input_dim,
            output_dim=self.hparams.output_dim,
            config=_head_callable._config_template(**self.hparams.head_config),
        )

    def forward(self, backbone_features: torch.Tensor) -> torch.Tensor:
        # B x L x T
        if not self.hparams.share_head_weights:
            # B x T X Output
            y_hat = torch.cat(
                [h(backbone_features[:, :, i]).unsqueeze(1) for i, h in enumerate(self.head)],
                dim=1,
            )
        else:
            # https://discuss.pytorch.org/t/how-to-pass-a-3d-tensor-to-linear-layer/908/6
            # B x T x L -> B x T x Output
            y_hat = self.head(backbone_features.transpose(2, 1))

        # applying weights to each tree and summing up
        # ETA
        y_hat = y_hat * self.eta.reshape(1, -1, 1)
        # summing up
        y_hat = y_hat.sum(dim=1)

        if self.hparams.task == "regression":
            y_hat = y_hat + self.T0
        return y_hat


class GateModulationFeatureUnitModel(TablFramework):
    def __init__(self, config: DictConfig, **kwargs):
        self.num_tasks = config.num_tasks
        self.n_heads = config.n_heads
        self.residual_scale = config.residual_scale
        super().__init__(config, **kwargs)

    @property
    def backbone(self):
        return self._backbone

    @property
    def embedding_layer(self):
        return self._embedding_layer

    @property
    def head(self):
        return self._head

    def _build_network(self):
        # Backbone
        self._backbone = GateModulationFeatureUnitBackbone(
            num_tasks = self.num_tasks,
            n_heads= self.n_heads,
            residual_scale = self.residual_scale,
            n_continuous_features=self.hparams.continuous_dim,
            cat_embedding_dims=self.hparams.embedding_dims,
            gflu_stages=self.hparams.gflu_stages,
            gflu_dropout=self.hparams.gflu_dropout,
            num_trees=self.hparams.num_trees,
            tree_depth=self.hparams.tree_depth,
            tree_dropout=self.hparams.tree_dropout,
            binning_activation=self.hparams.binning_activation,
            feature_mask_function=self.hparams.feature_mask_function,
            batch_norm_continuous_input=self.hparams.batch_norm_continuous_input,
            chain_trees=self.hparams.chain_trees,
        )
        # Embedding Layer
        self._embedding_layer = self._backbone._build_embedding_layer()
        # Head
        self._head = CustomHead(self.backbone.output_dim, self.hparams)
        # Heads (for multi-task)
        self._heads = nn.ModuleList(
            [CustomHead(self.backbone.output_dim, self.hparams) for _ in range(self.hparams.num_tasks)]
            )

    def data_aware_initialization(self, datamodule):
        if self.hparams.task == "regression":
            logger.info("Data Aware Initialization of T0")
            # Need a big batch to initialize properly
            alt_loader = datamodule.train_dataloader(batch_size=self.hparams.data_aware_init_batch_size)
            batch = next(iter(alt_loader))
            self.head.T0.data = torch.mean(batch["target"], dim=0)
