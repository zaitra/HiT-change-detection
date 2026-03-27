from functools import reduce
from collections import defaultdict
import torch
import segmentation_models_pytorch.metrics as metrics


class Stats:
    types = ['iou', 'f1', 'prec', 'rec', 'mcc']

    def __init__(self, n_classes, foreground_idx=1, ignore_index=-1, acc=False):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.foreground_idx = foreground_idx
        self.report_accuracy = acc

        self.data = defaultdict(int)
    
    def reset(self):
        self.data = defaultdict(int)

    def update(self, logits, target):
        
        pred = logits.argmax(dim=1)
        
        tp, fp, fn, tn = metrics.get_stats(pred, target,
                                           mode='multiclass',
                                           num_classes=self.n_classes,
                                           ignore_index=self.ignore_index
                                           )

        self.data['tp'] += tp.sum(dim=0).long()
        self.data['fp'] += fp.sum(dim=0).long()
        self.data['fn'] += fn.sum(dim=0).long()
        self.data['tn'] += tn.sum(dim=0).long()
    
    def compute(self):
        stats = {}

        for name, f in zip(['iou', 'f1', 'prec', 'rec'], [metrics.iou_score, metrics.f1_score, metrics.precision, metrics.recall]):
            r = f(self.data['tp'], self.data['fp'], self.data['fn'], self.data['tn'], reduction='none')
            stats[name] = {
                'micro': f(self.data['tp'], self.data['fp'], self.data['fn'], self.data['tn'], reduction='micro'),
                'macro': r.mean(),
                **{f'{i}': v for i, v in enumerate(r)}
            }

        stats['mcc'] = self._mcc_foreground()

        if self.report_accuracy:
            stats['accuracy'] = self._accuracy()

        return stats

    # def _mcc_foreground(self):
    #     class_idx = self.foreground_idx
    #     tp, tn, fp, fn = self.data['tp'][class_idx], self.data['tn'][class_idx], self.data['fp'][class_idx], self.data['fn'][class_idx]
    #     numerator = (tp * tn) - (fp * fn)
    #     denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))**0.5
    #     mcc = 0.0 if torch.isnan(denom) or denom == 0.0 else numerator / denom
    #     return mcc

    def _accuracy(self):
        correct = self.data['tp'].sum() + self.data['tn'].sum()
        total = correct + self.data['fp'].sum() + self.data['fn'].sum()
        return (correct / total).item()

    def _mcc_foreground(self):
        class_idx = self.foreground_idx
        tp = self.data['tp'][class_idx].double()
        tn = self.data['tn'][class_idx].double()
        fp = self.data['fp'][class_idx].double()
        fn = self.data['fn'][class_idx].double()

        numerator = (tp * tn) - (fp * fn)
        denom = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        # add eps to avoid nan/inf
        eps = 1e-8
        mcc = numerator / (denom + eps)
        # clamp to [-1, 1] for sanity
        if not torch.isnan(mcc).item() and mcc.item() > 2.0:
            print("MCC greater than 2.0 detected, clamping to 1.0") 
        mcc = torch.clamp(mcc, min=-1.0, max=1.0)
        return mcc.item() 

    def __add__(self, other):
        new_stats = Stats(self.n_classes, self.foreground_idx, self.ignore_index)

        for k in self.data:
            new_stats.data[k] = self.data[k] + other.data[k]

        return new_stats
    
