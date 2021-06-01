from functools import partial
from typing import Callable

import torch.nn as nn
import torch.nn.functional as F

from swyft.networks.resnet import BatchNorm1dWithChannel, LinearWithChannel, ResidualNet
from swyft.networks.tail import GenericTail


class EmbeddingWithChannel(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        input_dim: int,
        output_dim: int,
        activation: Callable = F.relu,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        self.linear = LinearWithChannel(channel_dim, input_dim, output_dim)
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = BatchNorm1dWithChannel(channel_dim, output_dim)

    def forward(self, input):
        x = self.linear(input)
        x = self.activation(x)
        if self.use_batch_norm:
            x = self.bn(x)
        return x


def make_resenet_tail(
    num_observation_features: int,
    parameter_list,
    hidden_features: int,
    num_blocks: int,
    online_z_score_obs: bool = True,
    online_z_score_par: bool = True,
    dropout_probability: float = 0.0,
    activation: Callable = F.relu,
    use_batch_norm: bool = True,
):
    get_ratio_estimator = partial(
        ResidualNet,
        out_features=1,
        hidden_features=hidden_features,
        num_blocks=num_blocks,
        activation=activation,
        dropout_probability=dropout_probability,
        use_batch_norm=use_batch_norm,
    )
    # # This one "feels better" although there is no evidence of its superiority
    # get_observation_embedding = partial(
    #     EmbeddingWithChannel,
    #     output_dim=hidden_features,
    #     activation=activation,
    #     use_batch_norm=use_batch_norm,
    # )
    # get_parameter_embedding = partial(
    #     EmbeddingWithChannel,
    #     output_dim=hidden_features,
    #     activation=activation,
    #     use_batch_norm=use_batch_norm,
    # )
    # This one had fewer NaNs for sampling, Although that could have been a good thing? sharper peaks?
    get_observation_embedding = partial(
        ResidualNet,
        out_features=hidden_features,
        hidden_features=hidden_features,
        num_blocks=1,
        activation=activation,
        dropout_probability=dropout_probability,
        use_batch_norm=use_batch_norm,
    )

    get_parameter_embedding = None

    return GenericTail(
        num_observation_features,
        parameter_list,
        get_ratio_estimator,
        get_observation_embedding,
        get_parameter_embedding,
        online_z_score_obs,
        online_z_score_par,
    )
