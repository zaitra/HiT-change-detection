import argparse
import tqdm
import glob
from collections import defaultdict
import yaml
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from model.lightning_wrapper import LightningWrapper

from data.sttorm import SttormDataset
from data.augmentations import augmentation_from_dict
from train import get_model_from_cfg
import yaml


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--output', type=str, default='./output')
parser.add_argument('--save-input', action='store_true')
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

with open(args.config, 'r') as f:
    cfg = yaml.safe_load(f)

NAME = args.name
OUTPUT_PATH = args.output
SAVE_INPUT = args.save_input
DEVICE = args.device
ckpt = args.ckpt

os.makedirs(f"{OUTPUT_PATH}/{NAME}", exist_ok=True)
if SAVE_INPUT:
    os.makedirs(f"{OUTPUT_PATH}/input", exist_ok=True)

state_dict = torch.load(ckpt)['state_dict']
state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}

net = PrithviModel(n_classes=cfg['model']['args']['n_classes'], **cfg['model']['unet']['args'])
net.load_state_dict(state_dict)
net.evel()
net.to(DEVICE)

cfg['data']['args']['continous_labels'] = False
test_transforms = augmentation_from_dict(cfg['data']['augmentations']['test'])
test_set = SttormDataset(cfg['data']['test']['path'], csv_path=cfg['data']['test']['csv'], transforms=test_transforms, **cfg['data']["args"])

outs = {}
max_xy = {}

for i, d in tqdm.tqdm(enumerate(test_set), total=len(test_set)):
    sample = test_set.data.iloc[i].to_dict()
    if sample['event'] not in outs:
        outs[sample['event']] = {i: defaultdict(list) for i in range(5)}
        max_xy[sample['event']] = (0,0)

    d['after'] = d['after'].unsqueeze(0).to(DEVICE)  
    d['before'] = [x.unsqueeze(0).to(DEVICE) for x in d['before']]
    d['label'] = d['label'].unsqueeze(0).to(DEVICE).long()
    empty_target = torch.zeros_like(d['label']).to(DEVICE).long()
    to_ignore = d['label'] == -1
    empty_target[to_ignore] = -1

    max_xy[sample['event']] = (
        max(max_xy[sample['event']][0], sample['x'] + 256),
        max(max_xy[sample['event']][1], sample['y'] + 256)
    )

    em = None # torch.zeros((1, 512, 8, 8)).to(DEVICE)
    with torch.no_grad():
        if "Baseline" in NAME:
            for i in range(len(d['before'])-1):
                out = net([d['before'][i]], d['before'][i+1])
                outs[sample['event']][i+1][(sample['x'], sample['y'])] = [out, d['before'][i], empty_target]

            out = net([d['before'][-1]], d['after'])
            outs[sample['event']][4][(sample['x'], sample['y'])] = [out, d['after'], d['label']]
        else:
            em = None
            _, em = net(d['before'][0], em, return_cls=False)
            for i in range(1,len(d['before'])):
                out, em = net(d['before'][i], em, return_cls=False)
                outs[sample['event']][i][(sample['x'], sample['y'])] = [out, d['before'][i], empty_target]
            out, _ = net(d['after'], em, return_cls=False)
            outs[sample['event']][4][(sample['x'], sample['y'])] = [out, d['after'], d['label']]

for event in outs.keys():
    H = max_xy[event][0]
    W = max_xy[event][1]

    r,c = H // 256, W // 256
    
    for i in range(1, 5):
        output = np.zeros((H, W, 3), dtype=np.uint8)
        orig   = np.zeros((H, W, 3), dtype=np.uint8)
        for (x, y), (out, after, lbl) in outs[event][i].items():
            o = out.cpu().numpy()[0]
            o = np.argmax(o, axis=0)

            a = after.cpu()[0,[2,1,0]].permute(1,2,0).numpy()
            l = lbl.cpu().numpy()[0]

            o_img = np.zeros((256,256, 3)) 
            o_img[np.logical_and((l == 1), (o == 1))] = [100, 255, 100]
            o_img[np.logical_and((l == 1), (o == 0))] = [100, 100, 255]
            o_img[np.logical_and((l == 0), (o == 1))] = [255, 100, 100]
            o_img[l == -1] = 0

            output[x:x+256, y:y+256] = o_img
            orig[x:x+256, y:y+256] = np.clip(a * 2 * 255, 0, 255).astype(np.uint8)
        
        sum_0 = np.sum(orig, axis=-1) 
        w = int(np.where(sum_0[0] == 0)[0][0])
        h = int(np.where(sum_0[:, 0] == 0)[0][0])

        plt.imsave(f"{OUTPUT_PATH}/{NAME}/{event}_{i}.png", output[:h, :w])

        if SAVE_INPUT and os.path.exists(f"{OUTPUT_PATH}/input/{event}_{i}.png") is False:
            plt.imsave(f"{OUTPUT_PATH}/input/{event}_{i}.png", orig[:h, :w])
        
