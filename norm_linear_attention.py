# https://github.com/Doraemonzzz/transnormer-v2-pytorch/blob/main/transnormer_v2/norm_linear_attention.py

import logging
import os
import sys

import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class NormAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_heads,
        act_fun="elu",
        uv_act_fun="swish",
        norm_type="layernorm",
        causal=False,
    ):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, hidden_dim)
        self.k_proj = nn.Linear(embed_dim, hidden_dim)
        self.v_proj = nn.Linear(embed_dim, hidden_dim)
        self.u_proj = nn.Linear(embed_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, embed_dim)
        self.act = F.elu
        self.uv_act = F.silu
        self.num_heads = num_heads
        self.norm = RMSNorm(hidden_dim)
        self.causal = causal

        self.batch_first = False

    def forward(
        self,
        x,
        y=None,
        attn_mask=None,
    ):
        # x: b n d
        if y == None:
            y = x
        n = x.shape[-2]
        # linear map
        q = self.q_proj(x)
        u = self.u_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        # uv act
        u = self.uv_act(u)
        v = self.uv_act(v)
        # reshape
        q, k, v = map(lambda x: rearrange(x, '... n (h d) -> ... h n d', h=self.num_heads), [q, k, v])
        # act
        q = self.act(q)
        k = self.act(k)

        if self.causal:
            if (attn_mask == None):
                attn_mask = (torch.tril(torch.ones(n, n))).to(q)
            l1 = len(q.shape)
            l2 = len(attn_mask.shape)
            for _ in range(l1 - l2):
                attn_mask = attn_mask.unsqueeze(0)
            energy = torch.einsum('... n d, ... m d -> ... n m', q, k)
            energy = energy * attn_mask
            output = torch.einsum('... n m, ... m d -> ... n d', energy, v)
        else:
            kv = torch.einsum('... n d, ... n e -> ... d e', k, v)
            output = torch.einsum('... n d, ... d e -> ... n e', q, kv)
        # reshape
        output = rearrange(output, '... h n d -> ... n (h d)')
        # normalize
        output = self.norm(output)
        # gate
        output = u * output
        # outproj
        output = self.out_proj(output)

        return output