import os
import glob
import pandas as pd
from enum import IntEnum

import numpy as np
import torch
import rasterio
from rasterio.windows import Window

from data.multitemp_dataset import MultiTemporalDataset


class Labels(IntEnum):
    CLOUDS = 2
    CHANGE = 1
    NO_CHANGE = 0

class RavaenDataset(MultiTemporalDataset):

    def load_data(self, path, n_prev_images, **kwargs):

        
        data = []
        cols = self.columns + ['clouds']

        for e in os.listdir(path):
            label = glob.glob(f"{path}/{e}/changes/*.tif")[0]

            image_files = sorted([f for f in glob.glob(f"{path}/{e}/S2/*.tif") ])
            after = image_files.pop()
            before = image_files[-n_prev_images:]

            h, w = -1, -1
            with rasterio.open(label) as src:
                h, w = src.height, src.width
                arr = src.read(1)
            
            chunks = [(x,y) for x in range(0, h, self.patch_size)
                            for y in range(0, w, self.patch_size)]
            
            for x, y in chunks:
                arr_c = arr[x:x+self.patch_size, y:y+self.patch_size]
                change = np.sum(arr_c == 1)
                clouds = np.sum(arr_c == 2)

                data.append([e, x, y, before, after, label, change/arr_c.size, clouds/arr_c.size])


        return pd.DataFrame(data, columns=cols)

    def load_sample(self, event, x, y, before, after, label, clouds, **kwargs):
        # col first
        window = Window(y, x, self.patch_size, self.patch_size)

        def load(path, label=False):
            bands = self.bands if not label else [1] 
            if len(bands) > 1:
                bands = [b + 1 for b in bands]

            with rasterio.open(path) as src: 
                data = src.read(bands, window=window)
                h, w = data.shape[1:3]
            
            value = self.ignore_index if label else 0 
            tile = np.ones((len(bands), self.patch_size, self.patch_size)) * value
            tile[:, :h, :w] = data

            if label:
                tile[tile == Labels.CLOUDS] = 0

            is_nan = np.isnan(tile)
            tile[is_nan] = value

            if label:
                tile = tile.squeeze(0)

            return torch.tensor(tile).float()

        l = load(label, label=True)
        a = load(after)
        b = [load(f) for f in before]

        l[torch.isnan(a[0]) | (a[0] == 0)] = self.ignore_index

        return dict(label=l, before=b, after=a)
