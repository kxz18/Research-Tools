#!/usr/bin/python
# -*- coding:utf-8 -*-
from .abs_trainer import TrainConfig
from .Seq2Seq_trainer import Seq2SeqTrainer


def create_trainer(model_type, model, train_loader, valid_loader, config):
    if model_type == 'seq2seq':
        Trainer = Seq2SeqTrainer
    else:
        raise NotImplementedError(f'Trainer for model type {model_type} not implemented!')
    return Trainer(model, train_loader, valid_loader, config)


