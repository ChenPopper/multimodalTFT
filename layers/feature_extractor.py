from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
from einops import rearrange


class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * var.clamp(min=self.eps).rsqrt() * self.g + self.b


class Block(nn.Module):
    def __init__(self, dim, dim_out,  # image_size,
                 k=1, s=1, p=0,
                 pooling_layer=None, act=nn.LeakyReLU()
                 ):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, kernel_size=k, stride=s, padding=p)
        # self.image_size = [
        #     int((image_size[0] + 2 * p - k) / s + 1),
        #     int((image_size[1] + 2 * p - k) / s + 1)
        # ]
        # self.norm = nn.LayerNorm([dim_out, self.image_size[0], self.image_size[1]])
        # self.norm = ChanLayerNorm(dim_out)
        self.norm = nn.BatchNorm2d(dim_out)
        self.act = act
        self.pooling = pooling_layer

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        if self.pooling is not None:
            x = self.pooling(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(
            self,
            dim,
            # image_size,
            conv_params,
            dim_out=None,
            cond_dim=None,
            pooling_layer=None,  # MaxPool2d or AdaptiveAvgPool2d
            act=nn.LeakyReLU()
    ):
        super().__init__()
        dim_out = dim_out or dim
        self.mlp = None
        self.pooling = pooling_layer

        if cond_dim is not None:
            self.mlp = nn.Sequential(
                nn.ReLU(),
                nn.Linear(cond_dim, dim_out * 2)
            )

        self.block1 = Block(
            dim, dim_out,  # image_size=image_size,
            k=conv_params[0], s=conv_params[1], p=conv_params[2],
            pooling_layer=pooling_layer, act=act
        )
        # self.image_size = self.block1.image_size
        # self.block2 = Block(
        #     dim_out, dim_out,  # image_size=self.image_size,
        #     k=conv_params[0], s=conv_params[1], p=conv_params[2], pooling_layer=pooling_layer, act=act
        # )
        # self.res_conv = nn.Conv2d(
        #     dim, dim_out,
        #     kernel_size=conv_params[0],
        #     stride=conv_params[1],
        #     padding=conv_params[2]
        # )  # if dim != dim_out else nn.Identity()

        self.res_conv = nn.Conv2d(
            dim, dim_out,
            kernel_size=1,
        ) if dim != dim_out else nn.Identity()
        self.downsampler = nn.MaxPool2d(
            kernel_size=conv_params[0], stride=conv_params[1], padding=conv_params[2]
        )

    def forward(self, x, cond=None):

        scale_shift = None

        assert not ((self.mlp is not None) ^ (cond is not None))

        if self.mlp is not None and cond is not None:
            cond = self.mlp(cond)
            cond = rearrange(cond, 'b c -> b c 1 1')
            scale_shift = cond.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        # h = self.block2(h)

        return h + self.downsampler(self.res_conv(x))


# class ResnetBlocks(nn.Module):
#     def __init__(
#         self,
#         dim,
#         *,
#         dim_in=None,
#         depth=1,
#         cond_dim=None
#     ):
#         super().__init__()
#         curr_dim = dim_in or dim
#
#         blocks = []
#         for _ in range(depth):
#             blocks.append(ResnetBlock(dim=curr_dim, dim_out=dim, cond_dim=cond_dim))
#             curr_dim = dim
#
#         self.blocks = nn.ModuleList(blocks)
#
#     def forward(self, x, cond=None):
#
#         for block in self.blocks:
#             x = block(x, cond=cond)
#
#         return x


class FeedForward(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dims: List[int],
            act: nn.Module = nn.LeakyReLU(),
            dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims.append(output_dim)
        fc = []
        for i in range(len(hidden_dims)):
            if i == 0:
                fc += [nn.Linear(input_dim, hidden_dims[0]), act]
            else:
                fc += [nn.Linear(hidden_dims[i - 1], hidden_dims[i]), act]
        fc.append(nn.Dropout(dropout))
        self.fc = nn.Sequential(*fc)

    def forward(self, x):
        return self.fc(x)


class FeaturePyramid(nn.Module):
    """
    The Feature Pyramid network
    """

    def __init__(
            self,
            # input_dims: Dict[List],
            conv_layers: List[nn.Module],
            resnet_layers: List[nn.Module],
            patching_layers: List[nn.Module],
            upsampler: nn.Module,  # nn.Upsample(scale_factor, mode='nearest')
            device=torch.device('cpu'),
            # dropout: float = 0.0,
            # skip_connection: bool = False,
    ):
        super().__init__()

        self.emb_1 = conv_layers[0].to(device)
        self.emb_2 = conv_layers[1].to(device)

        self.f_extractor_1 = resnet_layers[0].to(device)
        self.f_extractor_2 = resnet_layers[1].to(device)
        self.f_extractor_3 = resnet_layers[2].to(device)
        self.f_extractor_4 = resnet_layers[3].to(device)

        self.post_conv = conv_layers[2].to(device)
        self.patching_layers = [pl.to(device) for pl in patching_layers]
        self.upsampler = upsampler.to(device)

    def forward(self, x, y=None):
        # x shape -> (B, T, C, H, W)
        x = self.emb_1(x)
        x = self.f_extractor_1(x)
        if y is not None:
            y = self.emb_2(y)
            y = self.f_extractor_2(y)
            y = self.f_extractor_3(y)
            y = self.f_extractor_4(y)
            z = torch.cat([x, y], dim=-3)
            z = self.post_conv(z)
            z_patch = self.patching_layers[0](z)

            z_upsample_1 = self.upsampler(z)
            z_patch_1 = self.patching_layers[1](z_upsample_1)
            z_upsample_2 = self.upsampler(z_upsample_1)
            z_patch_2 = self.patching_layers[2](z_upsample_2)

            return torch.cat([z_patch, z_patch_1, z_patch_2], dim=1)
        else:
            x = self.post_conv(x)
            x = self.patching_layers[0](x)
            return x  # (B, T, H_out / patch_size * W_out / patch_size, E)


class CatFeatureEmbeder(nn.Module):

    def __init__(self, cardinatlities: List[torch.LongTensor], emb_dim):
        super().__init__()
        self.num_vars = len(cardinatlities)
        self.embeders = nn.ModuleList(
            [nn.Embedding(card, emb_dim) for card in cardinatlities]
        )

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            if x.dim() == 1:
                return [self.embeders[0](x)]
            else:
                assert x.shape[
                           -1] == self.num_vars, f"Expected the length of input list {self.num_vars}, got {x.shape[-1]}"
                features = [ele.squeeze() for ele in torch.chunk(x, x.shape[-1], dim=-1)]
        elif isinstance(x, List):
            assert len(x) == self.num_vars, f"Expected the length of input list {self.num_vars}, got {len(x)}"
            features = x
        else:
            raise RuntimeError('unsupported input type')
        return [embeder(feat) for embeder, feat in zip(self.embeders, features)]


class FeatureProjector(nn.Module):

    def __init__(
            self,
            feature_dims: int,
            embedding_dims: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dims, embedding_dims)

    def forward(self, x):
        return self.proj(x)


class GatedLinearUnit(nn.Module):
    def __init__(self, input_size, dim: int = -1, nonlinear: bool = True):
        super().__init__()
        self.dim = dim
        self.nonlinear = nonlinear
        self.fc = nn.Linear(input_size, input_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        value, gate = torch.chunk(x, chunks=2, dim=self.dim)
        if self.nonlinear:
            value = torch.tanh(value)
        gate = torch.sigmoid(gate)
        return gate * value


class GLU(nn.Module):
    # Gated Linear Unit
    def __init__(self, input_size):
        super(GLU, self).__init__()

        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class GatedResidualNetwork(nn.Module):
    def __init__(
            self,
            d_hidden: int,
            d_input: Optional[int] = None,
            d_output: Optional[int] = None,
            d_static: Optional[int] = None,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_input = d_input or d_hidden
        self.d_static = d_static or 0
        if d_output is None:
            self.d_output = self.d_input
            self.add_skip = False
        else:
            self.d_output = d_output
            if d_output != self.d_input:
                self.add_skip = True
                self.skip_proj = nn.Linear(
                    in_features=self.d_input,
                    out_features=self.d_output,
                )
            else:
                self.add_skip = False

        self.mlp = nn.Sequential(
            nn.Linear(
                in_features=self.d_input + self.d_static,
                out_features=self.d_hidden,
            ),
            nn.ELU(),
            nn.Linear(
                in_features=self.d_hidden,
                out_features=self.d_hidden,
            ),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=self.d_hidden,
                out_features=self.d_output,  # * 2,
            ),
            GatedLinearUnit(input_size=self.d_output, nonlinear=False),
        )
        self.layer_norm = nn.LayerNorm([self.d_output])

    def forward(
            self, x: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self.add_skip:
            skip = self.skip_proj(x)
        else:
            skip = x
        if self.d_static > 0 and c is None:
            raise ValueError("static variable is expected.")
        if self.d_static == 0 and c is not None:
            raise ValueError("static variable is not accepted.")
        if c is not None:
            x = torch.cat([x, c], dim=-1)
        x = self.mlp(x)
        x = self.layer_norm(x + skip)
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(
            self,
            d_hidden: int,
            num_vars: int,
            dropout: float = 0.0,
            add_static: bool = False,
    ) -> None:
        super().__init__()
        self.d_hidden = d_hidden
        self.num_vars = num_vars
        self.add_static = add_static

        self.weight_network = GatedResidualNetwork(
            d_hidden=self.d_hidden,
            d_input=self.d_hidden * self.num_vars,
            d_output=self.num_vars,
            d_static=self.d_hidden if add_static else None,
            dropout=dropout,
        )
        self.variable_networks = nn.ModuleList(
            [
                GatedResidualNetwork(d_hidden=d_hidden, dropout=dropout)
                for _ in range(num_vars)
            ]
        )

    def forward(
            self,
            variables: List[torch.Tensor],
            static: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flatten = torch.cat(variables, dim=-1)
        if static is not None:
            static = static.expand_as(variables[0])
        weight = self.weight_network(flatten, static)
        weight = torch.softmax(weight, dim=-1)

        var_encodings = [
            net(var) for var, net in zip(variables, self.variable_networks)
        ]
        var_encodings = torch.stack(var_encodings, dim=-1)

        var_encodings = torch.sum(var_encodings * weight.unsqueeze(-2), dim=-1)

        return var_encodings, weight


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class ImageEncoder(nn.Module):
    """
    The Image Encoder network
    """

    def __init__(
            self,
            # input_dims: Dict[List],
            conv_layers: List[nn.Module],
            resnet_layer: nn.Module,
            device=torch.device('cpu'),
    ):
        super().__init__()

        self.emb_1 = conv_layers[0].to(device)
        self.f_extractor_1 = resnet_layer.to(device)
        self.post_conv = conv_layers[1].to(device)

    def forward(self, x):
        # x shape -> (B, C, H, W)
        x = self.emb_1(x)
        x = self.f_extractor_1(x)
        x = self.post_conv(x)
        return x  # (B, C_out, H_out, E_out)