#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import namedtuple
from functools import wraps

import torch


OOMReturn = namedtuple('OOMReturn', ['fake_loss'])


def oom_decorator(forward):
    @wraps(forward)

    def deco_func(self, *args, **kwargs):
        try:
            output = forward(self, *args, **kwargs)
            return output
        except torch.cuda.OutOfMemoryError:
            output = sum([p.norm() for p in self.parameters()]) * 0.0
            return OOMReturn(output)
    
    return deco_func

