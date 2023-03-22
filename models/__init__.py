#!/usr/bin/python
# -*- coding:utf-8 -*-
from .Seq2Seq.model import Seq2Seq
from .pepdesign import PepDesign


def create_model(args):
    model_type = args.model_type
    if model_type == 'seq2seq':
        return Seq2Seq(hidden_size=args.hidden_size, depth=args.n_layers)
    else:
        raise NotImplementedError(f'Model type {model_type} not implemented!')