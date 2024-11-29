# Copyright (c) CASIA. All rights reserved.
import torch.nn as nn
from typing import Optional, Union, Callable
import torch
from torch import Tensor
from torch.nn import functional as F
import math
import timm
from scipy.optimize import linear_sum_assignment


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
K = 28 * 28
C = 512


def hungarian_matching(similarity_matrix):
    B, M, N = similarity_matrix.shape
    result = torch.zeros(B, M)
    
    for b in range(B):
        cost_matrix = -similarity_matrix[b].detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        result[b] = similarity_matrix[b, row_ind, col_ind]
    
    return result

class PositionEmbeddingSine(nn.Module):
    """
    2D Sine Postion Embedding.
    Copyed from UniAD.
    """

    def __init__(
        self,
        feature_size,
        num_pos_feats=128,
        temperature=10000,
        normalize=False,
        scale=None,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        not_mask = torch.ones((self.feature_size[0], self.feature_size[1]))  # H x W
        y_embed = not_mask.cumsum(0, dtype=torch.float32)
        x_embed = not_mask.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).flatten(0, 1)  # (H X W) X C
        return tensor + pos.to(tensor.device)

class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.backbone = timm.create_model("resnet18",pretrained=True,features_only=True)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self,x):

        self.backbone.eval()

        with torch.no_grad():
            self.feat_input = self.backbone(x)


        for i in [0,1,2,3]:
            self.feat_input[i] = nn.Upsample(size=(28, 28),mode='bicubic')(self.feat_input[i])


        # 将四个 Stage 的 feature 进行拼接
        # [bath_size, 512, 28, 28]
        x = torch.cat(self.feat_input[0:4],dim=1)

        # [batch_size, 512, 784]
        x = nn.Flatten(-2,-1)(x)

        # [batch_size, 784, 512]
        x = x.transpose(-2,-1)

        return x

class Customed_MultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device,
            dtype,
        )

        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = False

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs)
            )
            self.k_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(
                torch.empty(3 * embed_dim, **factory_kwargs)
            )
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()


class Encoder_layer(nn.TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        layers=4,
        bias: bool = True,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = Customed_MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )

        self.alpha = 0.81 * (
            (layers**4 * 4) ** (1 / 16)
        )  # (2*layers)**(1/4) If Only Encoder
        self.beta = 0.87 * (
            (layers**4 * 4) ** (-1 / 16)
        )  # (8*layers)**(-1/4) If Only Encoder

        # Modify the initialization parameters with reference to DeepNet
        nn.init.xavier_normal_(self.self_attn.q_proj_weight, gain=1)
        nn.init.xavier_normal_(self.self_attn.k_proj_weight, gain=1)

        nn.init.xavier_normal_(self.self_attn.v_proj_weight, gain=self.beta)
        nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=self.beta)
        nn.init.xavier_normal_(self.linear1.weight, gain=self.beta)
        nn.init.xavier_normal_(self.linear2.weight, gain=self.beta)

    # Self Attention Block
    def _sa_block(
        self, query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor]
    ) -> Tensor:

        x = self.self_attn(query, key, value, attn_mask=attn_mask)[0]
        return self.dropout1(x)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = query
        x = self.norm1(self.alpha * x + self._sa_block(query, key, value, src_mask))
        x = self.norm2(self.alpha * x + self._ff_block(x))
        return x


class Decoder_layer(nn.TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = True,
        norm_first: bool = False,
        device=None,
        dtype=None,
        layers=4,
        bias: bool = True,
    ) -> None:
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device,
            dtype,
        )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = Customed_MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.multihead_attn = Customed_MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )

        self.alpha = (3 * layers) ** (1 / 4)
        self.beta = (12 * layers) ** (-1 / 4)

        nn.init.xavier_normal_(self.self_attn.q_proj_weight, gain=1)
        nn.init.xavier_normal_(self.self_attn.k_proj_weight, gain=1)
        nn.init.xavier_normal_(self.multihead_attn.q_proj_weight, gain=1)
        nn.init.xavier_normal_(self.multihead_attn.k_proj_weight, gain=1)

        nn.init.xavier_normal_(self.self_attn.v_proj_weight, gain=self.beta)
        nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=self.beta)
        nn.init.xavier_normal_(self.multihead_attn.v_proj_weight, gain=self.beta)
        nn.init.xavier_normal_(self.multihead_attn.out_proj.weight, gain=self.beta)
        nn.init.xavier_normal_(self.linear1.weight, gain=self.beta)
        nn.init.xavier_normal_(self.linear2.weight, gain=self.beta)

    def forward(
        self,
        q: Tensor,
        k1: Tensor,
        v1: Tensor,
        k2: Tensor,
        v2: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = q
        x = self.norm1(self.alpha * x + self._first_block(x, k1, v1, mask))
        x = self.norm2(self.alpha * x + self._second_block(x, k2, v2, None))
        x = self.norm3(self.alpha * x + self._ff_block(x))
        return x

    def _first_block(
        self, x: Tensor, k1: Tensor, v1: Tensor, attn_mask: Tensor
    ) -> Tensor:
        x = self.self_attn(x, k1, v1, attn_mask=attn_mask)[0]
        return self.dropout1(x)

    def _second_block(
        self, x: Tensor, k2: Tensor, v2: Tensor, attn_mask: Tensor
    ) -> Tensor:
        x = self.multihead_attn(x, k2, v2, attn_mask=attn_mask)[0]
        return self.dropout2(x)


class ADformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # Sine Position Embedding
        self.pos_embed = PositionEmbeddingSine((28, 28), C // 2, normalize=True)

        # four-layers Encoder
        self.encoder_layer1 = Encoder_layer(
            d_model=C, nhead=8, batch_first=True, dropout=0, layers=4
        )
        self.encoder_layer2 = Encoder_layer(
            d_model=C, nhead=8, batch_first=True, dropout=0, layers=4
        )
        self.encoder_layer3 = Encoder_layer(
            d_model=C, nhead=8, batch_first=True, dropout=0, layers=4
        )
        self.encoder_layer4 = Encoder_layer(
            d_model=C, nhead=8, batch_first=True, dropout=0, layers=4
        )

        # four-layers Decoder
        self.decoder_layer1 = Decoder_layer(
            d_model=C, nhead=8, batch_first=True, dropout=0, layers=4
        )
        self.decoder_layer2 = Decoder_layer(
            d_model=C, nhead=8, batch_first=True, dropout=0, layers=4
        )
        self.decoder_layer3 = Decoder_layer(
            d_model=C, nhead=8, batch_first=True, dropout=0, layers=4
        )
        self.decoder_layer4 = Decoder_layer(
            d_model=C, nhead=8, batch_first=True, dropout=0, layers=4
        )

    def forward(self, x):

        # Add Position Embedding
        x = self.pos_embed(x)
        x = nn.modules.normalization.LayerNorm(C, device=x.device)(x)

        cnn_output = x

        # four-layers Encoder
        mask = gene_mask(x)
        x = self.encoder_layer1(x, x, x, mask)
        mask = gene_mask(x)
        x = self.encoder_layer2(x, x, x, mask)
        mask = gene_mask(x)
        x = self.encoder_layer3(x, x, x, mask)
        mask = gene_mask(x)
        x = self.encoder_layer4(x, x, x, mask)

        # four-layers Decoder
        mask = gene_mask(cnn_output)
        y = self.decoder_layer1(cnn_output, cnn_output, cnn_output, x, x, mask)
        mask = gene_mask(y)
        y = self.decoder_layer2(y, y, y, x, x, mask)
        mask = gene_mask(y)
        y = self.decoder_layer3(y, y, y, x, x, mask)
        mask = gene_mask(y)
        y = self.decoder_layer4(y, y, y, x, x, mask)

        return y


# Calculate the Mask through the cosine similarity between patches
def gene_mask(x):
    x = x.reshape([-1, 28 * 28, 512])
    x = F.normalize(x, dim=-1)

    mask = torch.matmul(x, x.transpose(-1, -2).contiguous())

    # use the median as the threshold
    thershold = mask.median()
    mask = (
        mask.float()
        .masked_fill(mask < thershold, float(-10000.0))
        .masked_fill(mask >= thershold, float(0.0))
    )

    # repeat 8 heads
    mask = torch.repeat_interleave(mask, 8, dim=0)

    return mask
