# noqa W605
from typing import Any, Dict, Tuple

import torch
from torch import nn

class Embedding1dLayer(nn.Module):
    """
    Enables different values in a categorical features to have different embeddings
    """

    def __init__(
        self,
        continuous_dim: int,
        categorical_embedding_dims: Tuple[int, int],
        embedding_dropout: float = 0.0,
        batch_norm_continuous_input: bool = False,
    ):
        super(Embedding1dLayer, self).__init__()
        self.continuous_dim = continuous_dim
        self.categorical_embedding_dims = categorical_embedding_dims
        self.batch_norm_continuous_input = batch_norm_continuous_input

        # Embedding layers
        self.cat_embedding_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in categorical_embedding_dims])
        if embedding_dropout > 0:
            self.embd_dropout = nn.Dropout(embedding_dropout)
        else:
            self.embd_dropout = None
        # Continuous Layers
        if batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(continuous_dim)

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        assert "continuous" in x or "categorical" in x, "x must contain either continuous and categorical features"
        # (B, N)
        continuous_data, categorical_data = x.get("continuous", torch.empty(0, 0)), x.get(
            "categorical", torch.empty(0, 0)
        )
        assert categorical_data.shape[1] == len(
            self.cat_embedding_layers
        ), "categorical_data must have same number of columns as categorical embedding layers"
        assert (
            continuous_data.shape[1] == self.continuous_dim
        ), "continuous_data must have same number of columns as continuous dim"
        embed = None
        if continuous_data.shape[1] > 0:
            if self.batch_norm_continuous_input:
                embed = self.normalizing_batch_norm(continuous_data)
            else:
                embed = continuous_data
            # (B, N, C)
        if categorical_data.shape[1] > 0:
            categorical_embed = torch.cat(
                [
                    embedding_layer(categorical_data[:, i])
                    for i, embedding_layer in enumerate(self.cat_embedding_layers)
                ],
                dim=1,
            )
            # (B, N, C + C)
            if embed is None:
                embed = categorical_embed
            else:
                embed = torch.cat([embed, categorical_embed], dim=1)
        if self.embd_dropout is not None:
            embed = self.embd_dropout(embed)
        return embed
