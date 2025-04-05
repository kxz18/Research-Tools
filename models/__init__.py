#!/usr/bin/python
# -*- coding:utf-8 -*-
from .sin_example_mlp import SinMLP

import utils.register as R

def create_model(config: dict, **kwargs):
    return R.construct(config, **kwargs)