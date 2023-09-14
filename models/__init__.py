#!/usr/bin/python
# -*- coding:utf-8 -*-

import utils.register as R

def create_model(config: dict, **kwargs):
    return R.construct(config, **kwargs)