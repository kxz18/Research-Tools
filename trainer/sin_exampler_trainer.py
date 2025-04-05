#!/usr/bin/python
# -*- coding:utf-8 -*-
from utils import register as R
from .abs_trainer import Trainer


@R.register('SinTrainer')
class SinTrainer(Trainer):

    def __init__(self, model, train_loader, valid_loader, config: dict, save_config: dict):
        super().__init__(model, train_loader, valid_loader, config, save_config)

    ########## Override start ##########

    def train_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx)

    def valid_step(self, batch, batch_idx):
        return self.share_step(batch, batch_idx, val=True)
    
    def _train_epoch_end(self, device):
        dataset = self.train_loader.dataset
        if hasattr(dataset, 'update_epoch'):
            dataset.update_epoch()
        return super()._train_epoch_end(device)

    ########## Override end ##########

    def share_step(self, batch, batch_idx, val=False):
        loss = self.model(*batch)

        log_type = 'Validation' if val else 'Train'

        self.log(f'loss/{log_type}', loss, batch_idx, val)

        if not val:
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.log('lr', lr, batch_idx, val)

        return loss