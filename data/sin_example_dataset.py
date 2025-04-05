#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
import torch

import utils.register as R


@R.register('SinDataset')
class SinDataset(torch.utils.data.Dataset):
    def __init__(self, length=10000):
        super().__init__()
        self.length = length

    def __getitem__(self, index):
        x = index / self.length * 2 * math.pi
        x = torch.tensor(x, dtype=torch.float).unsqueeze(0)
        y = torch.sin(x)
        return x, y
    
    def __len__(self):
        return self.length