import tqdm
import argparse
import yaml
import torch
import os
import numpy as np
from model.lightning_wrapper import LightningWrapper
from model.stats import Stats

from data.sttorm import SttormDataset
from data.augmentations import augmentation_from_dict
from train import get_model_from_cfg

from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

NAME = args.name
DEVICE = args.device
ckpt = args.ckpt

state_dict = torch.load(ckpt)['state_dict']
state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}

net = PrithviModel(n_classes=cfg['model']['args']['n_classes'], **cfg['model']['unet']['args'])
net.load_state_dict(state_dict)
net.evel()
net.to(DEVICE)


cfg['data']['args']['continous_labels'] = False

test_transforms = augmentation_from_dict(cfg['data']['augmentations']['test'])
test_set = SttormDataset(cfg['data']['test']['path'], csv_path=cfg['data']['test']['csv'], transforms=test_transforms, **cfg['data']["args"])

data = test_set.data

stats = {
    event: {
        'before': {i+1 : Stats(n_classes=2, foreground_idx=1, ignore_index=-1, acc=True) for i in range(test_set.n_prev_images)},
        'after':  {i+1 : Stats(n_classes=2, foreground_idx=1, ignore_index=-1, acc=True) for i in range(test_set.n_prev_images)},
    }
    for event in test_set.data.event.unique()
}

for event in test_set.data.event.unique():
    event_data = data.copy()
    event_data = event_data[event_data['event'] == event].reset_index(drop=True)
    test_set.data = event_data

    b_stats = stats[event]['before']
    a_stats = stats[event]['after']

    for i, d in tqdm.tqdm(enumerate(test_set), total=len(test_set), desc=f"Evaluating event {event}"):
        sample = test_set.data.iloc[i].to_dict()

        d['after'] = d['after'].unsqueeze(0).to(DEVICE)  
        d['before'] = [x.unsqueeze(0).to(DEVICE) for x in d['before']]
        d['label'] = d['label'].unsqueeze(0).to(DEVICE).long()
        empty_target = torch.zeros_like(d['label']).to(DEVICE).long()
        to_ignore = d['label'] == -1
        empty_target[to_ignore] = -1

        em = None # torch.zeros((1, 512, 8, 8)).to(DEVICE)
        with torch.no_grad():
            if "Baseline" in ckpt:
                for i in range(len(d['before'])-1):
                    out = net([d['before'][i]], d['before'][i+1])
                    out2 = net([d['before'][i]], d['after'])
                    b_stats[i+1].update(out, empty_target)
                    a_stats[i+1].update(out2, d['label'])
                out = net([d['before'][-1]], d['after'])
                a_stats[len(d['before'])].update(out, d['label'])
            else:
                em = None
                _, em = net(d['before'][0], em, return_cls=False)
                for i in range(1,len(d['before'])):
                    out2, _ = net(d['after'], em, return_cls=False)
                    out, em = net(d['before'][i], em, return_cls=False)
                    b_stats[i].update(out, empty_target)
                    a_stats[i].update(out2, d['label'])
                out, _ = net(d['after'], em, return_cls=False)
                a_stats[len(d['before'])].update(out, d['label'])

summary_stats = {
        'before': {i+1 : Stats(n_classes=2, foreground_idx=1, ignore_index=-1, acc=True) for i in range(test_set.n_prev_images)},
        'after':  {i+1 : Stats(n_classes=2, foreground_idx=1, ignore_index=-1, acc=True) for i in range(test_set.n_prev_images)},
}

for event in stats:
    for i in range(1,5):
        summary_stats['before'][i] =  stats[event]['before'][i] + summary_stats['before'][i]
        summary_stats['after'][i]  =  stats[event]['after'][i]  + summary_stats['after'][i]


# -------------- PRINT RESULTS --------------

# Always print aggregated summary to stdout (no file writes)
def _print_table(title, headers, rows, align_right=None):
    print()
    print(title)
    print(tabulate(rows, headers=headers, tablefmt="github", floatfmt=".4f", numalign="decimal", stralign="left"))

# Build summary tables
before_cols = ["Frame No.", "TP", "FP", "FN", "TN"]
before_rows = []
for i in range(1, 4):
    s = summary_stats['before'][i]
    before_rows.append([i, int(s.data['tp'][1].item()), int(s.data['fp'][1].item()), int(s.data['fn'][1].item()), int(s.data['tn'][1].item())])

after_cols = ["Frame No.", "TP", "FP", "FN", "TN"]
after_rows = []
for i in range(1, 5):
    s = summary_stats['after'][i]
    after_rows.append([i, int(s.data['tp'][1].item()), int(s.data['fp'][1].item()), int(s.data['fn'][1].item()), int(s.data['tn'][1].item())])

_print_table("Summary Statistics (aggregated across events) - Before", before_cols, before_rows, align_right=[True]*5)
_print_table("Summary Statistics (aggregated across events) - After", after_cols, after_rows, align_right=[True]*5)
        
# Always print per-event results to stdout (no file writes)
cols = ["Event", "Frame No.", "Acc Before", "F1 Before", "Acc After", "F1 After"]
rows = []
for event in stats:
    for idx in stats[event]['before']:
        if idx == 4:
            b_res = {'accuracy': float('nan'), 'f1': {'1': float('nan')}}
        else:
            b_res = stats[event]['before'][idx].compute()
        a_res = stats[event]['after'][idx].compute()
        def _fmt(x):
            try:
                return float(x)
            except Exception:
                return float('nan')
        rows.append([event, idx, _fmt(b_res['accuracy']), _fmt(b_res['f1']['1']), _fmt(a_res['accuracy']), _fmt(a_res['f1']['1'])])

_print_table("Per-event Results", cols, rows, align_right=[False, True, True, True, True, True])



