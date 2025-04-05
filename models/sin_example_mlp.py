#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

import utils.register as R


@R.register('SinMLP')
class SinMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super().__init__()
        assert n_layers >= 2, 'At least an input linear and an output linear are needed'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        modules = []
        modules.append(nn.Linear(input_size, hidden_size))
        modules.append(nn.SiLU())
        for _ in range(self.n_layers - 2):
            modules.append(nn.Linear(hidden_size, hidden_size))
            modules.append(nn.SiLU())
        modules.append(nn.Linear(hidden_size, output_size))
        self.module_list = nn.ModuleList(modules)
        self.no_use_param = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, y):
        h = x
        for m in self.module_list:
            h = m(h)
        h = h
        
        # calculate loss
        loss = F.mse_loss(h, y)
        return loss