#%%
import sys
sys.path.append("/home/dkyselica/work/sigma-knowledge-distillation/")

#%%
import numpy as np

import torch
from src.utils import SENTINEL2_BANDS
from data.multitemp_dataset import MultiTemporalDataset


BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B11", "B12", "B8A"]
SENTINEL_IDX = [SENTINEL2_BANDS.index(b) for b in BANDS]

class FLOGA(MultiTemporalDataset):
    FROM_KAUS = True 

    def load_data(self, path, n_prev_images, **kwargs):
        print("Create dataset using FLOGA repository")
        pass

    def load_sample(self, event, before, after, label,  **kwargs):
        # col first
        out_bands = []
        in_bands = []
        for i,b in enumerate(self.bands):
            if b in SENTINEL_IDX:
                out_bands.append(i)
                in_bands.append(SENTINEL_IDX.index(b))

        def load(path, label=False):
            if label and not isinstance(path, str):
                return torch.ones((self.patch_size, self.patch_size), dtype=torch.float32) 
            
            path = self._translate_path_from_kaus(path) if self.FROM_KAUS else path

            data = np.load(path)
            
            value = self.ignore_index if label else 0 
            if not label:
                tile = np.ones((len(self.bands), self.patch_size, self.patch_size)) * value
                tile[out_bands] = data[in_bands]
                is_nan = np.isnan(data[0])
                tile[:, is_nan] = value
            else:
                tile = data
                tile[tile > 1] = 0 # remove Clouds

            is_nan = np.isnan(tile)
            tile[is_nan] = value

            return torch.tensor(tile).float()
        
        l = load(label, label=True)
        a = load(after)
        b = [load(f) for f in before]
        is_nan = torch.logical_or(torch.isnan(a[in_bands[0]]), a[in_bands[0]] == 0)
        l[is_nan] = self.ignore_index
        
        return dict(label=l, before=b, after=a)

    def _translate_path_from_kaus(self, path):
        return path.replace('md1/', '')


