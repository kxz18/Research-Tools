#!/usr/bin/python
# -*- coding:utf-8 -*-
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


def kl_loss(model_mu, model_log_var, prior_mu=0.0, prior_std=1.0):
    '''
        p is the model generated distribution, N(mu_1, sigma_1), q is the prior N(mu_2, sigma_2)
        KL(p, q) = log(sigma_2/sigma_1) +(sigma_1^2 + (mu_1 - mu_2)^2) / (2*sigma_2^2) - 1/2
                 = -0.5 * (1.0 + log(sigma_1^2) - log(sigma_2^2) - (sigma_1^2 + (mu_1 - mu_2)^2) / sigma_2^2)
    '''
    model_var = torch.exp(model_log_var)
    prior_var = prior_std ** 2
    delta_mu = model_mu - prior_mu
    if isinstance(prior_std, float): prior_std = torch.tensor(prior_std, device=model_log_var.device)
    kl = -0.5 * torch.sum((1.0 + model_log_var - 2 * torch.log(prior_std) - (delta_mu * delta_mu + model_var) / prior_var))
    return kl