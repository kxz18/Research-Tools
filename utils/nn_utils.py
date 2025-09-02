#!/usr/bin/python
# -*- coding:utf-8 -*-
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_sum


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


def expand_like(src, tgt):
    src = src.reshape(*src.shape, *[1 for _ in tgt.shape[len(src.shape):]]) # [..., 1, 1, ...]
    return src.expand_as(tgt)


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sin-Cos Positional Embedding
    """
    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim

    def forward(self, position_ids):
        device = position_ids.device
        position_ids = position_ids[None] # [1, N]
        indices = torch.arange(self.output_dim // 2, device=device, dtype=torch.float)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)
        embeddings = torch.einsum('bn,d->bnd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(-1, self.output_dim)
        return embeddings


class SinusoidalTimeEmbeddings(nn.Module):
    '''
        sin(1*t*2pi), sin(2*t*2pi), ...,
        cos(1*t*2pi), cos(2*t*2pi)
    '''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        if self.dim == 1: # no projection
            return time.reshape(-1,1)
        device = time.device
        half_dim = self.dim // 2
        freq = 2 * torch.arange(half_dim, device=device) * math.pi
        t = freq * time[..., None]
        embeddings = torch.cat((t.sin(), t.cos()), dim=-1)
        return embeddings
