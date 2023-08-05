import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import nn
from torch import Tensor

from norm_linear_attention import NormAttention


class TransNormerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d_model, nhead, *_ = args
        hidden_dim = 512
        self.self_attn = NormAttention(d_model, hidden_dim, nhead)

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x,
                           attn_mask=attn_mask,)[0]
        return self.dropout1(x)