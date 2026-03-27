import os
import glob

import pandas as pd
import numpy as np
import torch

from data.multitemp_dataset import MultiTemporalDataset

ALL_BANDS = list(range(13))

class SttormDataset(MultiTemporalDataset):

    def load_data(self, path, n_prev_images, **kwargs) -> pd.DataFrame:

        data = []
        cols = self.columns 

        for e_path in sorted(glob.glob(f"{path}/*/")):
            e = os.path.basename(e_path[:-1])

            if not os.path.exists(f"{path}/{e}/all_bands"):
                p_paths = [(f, e_path + f) for f in os.listdir(f"{e_path}") if f.isnumeric()]
            else:
                p_paths = [(0, e_path)]
            
            for p, p_path in p_paths:
                images = sorted(glob.glob(f"{p_path}/all_bands/*.npy"))
                label_file = f"{p_path}/mask.npy"
                label = np.load(label_file)
                change = np.sum(label == 1) / (label != self.ignore_index).sum()

                after = images.pop()
                before = images[-n_prev_images:]
                
                for x in range(0, label.shape[0], self.patch_size):
                    for y in range(0, label.shape[1], self.patch_size):
                        chunk = label[x:x+self.patch_size, y:y+self.patch_size]
                        change = np.sum(chunk == 1) / chunk.size
                        data.append([f"{e}_{p}", x, y, before, after, label_file, change])

        return pd.DataFrame(data, columns=cols)
    
    def load_sample(self, event, x, y, before, after, label, **kwargs):

        def load(path, label=False):
            bands = self.bands if not label else [0]
            value = self.ignore_index if label else 0
            p = self.patch_size

            img = np.atleast_3d(np.load(path)[x:x+p, y:y+p])
            img = img[:,:,bands]

            tile = np.ones((p, p,len(bands))) * value
            tile[:img.shape[0], :img.shape[1]] = img 

            tile = torch.tensor(tile)
            tile = tile.permute(2, 0, 1)

            if label:
                tile = tile.squeeze(0)

            return tile
        
        l = load(label, label=True)
        a = load(after)
        b = [load(f) for f in before]

        # l[torch.isnan(a[0]) | (a.sum(dim=0) == 0)] = self.ignore_index


        return dict(label=l, before=b, after=a)
