from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from models.dice_loss import DiceLoss
from models.stats import Stats
from models.prithvi import PrithviModel

class LightningWrapper(pl.LightningModule):

    def __init__(self,
                 model: PrithviModel,
                 n_classes:int =2,
                 lr: float = 1e-4,
                 ignore_index: int = -1,
                 full_skip: bool= True,
                 max_epochs: int = 1000,
                 version: int = 1,
                 continuous_labels: bool = False
                 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])

        self.version = version

        self.model = model

        self.lr = lr
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.full_skip = full_skip
        self.max_epochs = max_epochs
        self.continuous_labels = continuous_labels

        self.dice_loss = DiceLoss(n_classes=n_classes, ignore_index=ignore_index, mode='micro')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.stats = {
            "train": Stats(n_classes, foreground_idx=1, ignore_index=ignore_index),
            "val": Stats(n_classes, foreground_idx=1, ignore_index=ignore_index),
            "test": defaultdict(lambda : Stats(n_classes, foreground_idx=1, ignore_index=ignore_index))
        }

        if self.continuous_labels:
            self.stats['continuous'] = {
                "train": Stats(n_classes, foreground_idx=1, ignore_index=ignore_index),
                "val": Stats(n_classes, foreground_idx=1, ignore_index=ignore_index),
                "test": defaultdict(lambda : Stats(n_classes, foreground_idx=1, ignore_index=ignore_index))
            }   

        self.mode = 'train'

    def forward(self, batch):

        B = batch['after'].shape[0]
        if not self.continuous_labels:
            target = batch['label'].long()
            em = torch.zeros((B, self.model.em_shape[0], 8,8), device=batch['after'].device)
        else:
            target = batch['label'][-1].long()
        loss = 0.
    

        cls_acc = torch.tensor([torch.nan])

        additional_outputs = {}

        
        if not self.model.baseline:
                
            outs, ems = self.model.forward_series(batch['before'], batch['after'], return_cls=False)

            continuous_accs = []
            for o, l in zip(outs[1:], batch['label']):
                l = l.long()
                loss += (self.ce_loss(o, l) + self.dice_loss(o, l)) 
                z = (o.argmax(dim=1) == l).float().mean()
                continuous_accs.append(z)
            
            em = ems[-1]
            out = outs[-1]

            additional_outputs['continuous_outs'] = outs[1:]
            additional_outputs['continuous_acc'] = torch.stack(continuous_accs).mean()
        else:
            if self.model.num_frames == 5:
                target = batch['label'][-1].long()
                out = self.model(batch['before'], batch['after'])
                loss += self.ce_loss(out, target)
                loss += self.dice_loss(out, target)
                ems = []
            elif self.model.num_frames == 2:
                continuous_accs = []
                outs = []
                for i in range(1,len(batch['before'])-1):
                    o = self.model([batch['before'][i-1]], batch['before'][i])
                    l = batch['label'][i-1].long()
                    loss += self.ce_loss(o, l) + self.dice_loss(o, l)
                    z = (o.argmax(dim=1) == l).float().mean()
                    continuous_accs.append(z)
                    outs.append(o)

                out   = self.model([batch['before'][-1]], batch['after'])
                target = batch['label'][-1].long()
                loss += self.ce_loss(out, target)
                loss += self.dice_loss(out, target)
                outs.append(out)
                continuous_accs.append((out.argmax(dim=1) == target).float().mean())

                additional_outputs['continuous_outs'] = outs
                additional_outputs['continuous_acc'] = torch.stack(continuous_accs).mean()

                ems = []

        acc = (out.argmax(dim=1) == target).float().mean()

        return {
            'cls_acc': cls_acc,
            'out': out,
            'ems': ems,
            'loss': loss,
            'acc': acc,
            **additional_outputs
        }

    def _log_metrics(self, results, label, prog_bar=["acc", "loss"]):

        for name, val in results.items():
            if isinstance(val, torch.Tensor) and val.numel() == 1:
                prog = name in prog_bar
                log_path = f"{label}/{name}"
                log_path = f"{self.mode}/{log_path}" if self.mode != 'train' else log_path
                on_step = name == 'loss'
                self.log(log_path, val.item(), on_step=on_step, on_epoch=True, prog_bar=prog)
            else:
                assert name not in ['acc', 'loss', 'cls_acc'], f"Some scalar metrics have not been logged {name} {val}"

    def training_step(self, batch, batch_idx):
        r = self.forward(batch)

        if self.continuous_labels:
            target = batch['label'][-1].long()
        else:
            target = batch['label'].long()

        self.stats['train'].update(r['out'], target)

        if self.continuous_labels:
            for o,l in zip(r['continuous_outs'][:-1], batch['label'][:-1]):
                self.stats['continuous']['train'].update(o, l.long()) 
    
        self._log_metrics(r, "train")

        return r['loss']
    
    
    def _val_test_step(self, batch, split='val', dataloader_idx=-1):
        r = self.forward(batch)

        if self.continuous_labels:
            target = batch['label'][-1].long()
        else:
            target = batch['label'].long()


        if split == 'test':
            s = self.stats[split][dataloader_idx]
            log_name = f"{split}/{dataloader_idx}"
        else:
            s = self.stats[split]
            log_name = split

        if self.continuous_labels:
            if split == 'test':
                cs = self.stats['continuous'][split][dataloader_idx]
            else:
                cs = self.stats['continuous'][split]
            for o,l in zip(r['continuous_outs'][:-1], batch['label'][:-1]):
                cs.update(o, l.long()) 

        s.update(r['out'], target)

        self._log_metrics(r, log_name)

        return r['loss']

    def validation_step(self, batch, batch_idx):
        return self._val_test_step(batch, split='val')
    
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._val_test_step(batch,  split='test', dataloader_idx=dataloader_idx)
    
    def _log_dict(self, data_dict, path=[]):
        for k, v in data_dict.items():
            cur_path = path + [k]
            if isinstance(v, dict):
                self._log_dict(v, cur_path)
            else:
                log_path = f'{"/".join(cur_path)}'
                log_path = f"{self.mode}/{log_path}" if self.mode != 'train' else log_path
                self.log(log_path, v, on_step=False, on_epoch=True)

    def on_train_epoch_end(self):
        stats = self.stats['train']
        self._log_dict(stats.compute(), path=['train'])
        stats.reset()
        
        if self.continuous_labels:
            c_stats = self.stats['continuous']['train']
            self._log_dict(c_stats.compute(), path=['continuous_train'])
            c_stats.reset()

    def on_validation_epoch_end(self):
        stats = self.stats['val']
        self._log_dict(stats.compute(), path=['val'])
        stats.reset()

        if self.continuous_labels:
            c_stats = self.stats['continuous']['val']
            self._log_dict(c_stats.compute(), path=['continuous_val'])
            c_stats.reset()

    def on_test_epoch_end(self):

        test_stats = None
        for d_idx, s in self.stats['test'].items():
            if test_stats is None:
                test_stats = s
            else:
                test_stats += s
            self._log_dict(s.compute(), path=['test', str(d_idx)])
        
        self._log_dict(test_stats.compute(), path=['test_merged'])        

        for d_idx, s in self.stats['test'].items():
            s.reset()
        
        if self.continuous_labels:
            for d_idx, s in self.stats['continuous']['test'].items():
                self._log_dict(s.compute(), path=['continuous_test', str(d_idx)])
                s.reset()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

