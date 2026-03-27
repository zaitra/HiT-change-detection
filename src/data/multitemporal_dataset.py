from copy import deepcopy
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from enum import StrEnum
import random
import ast

import numpy as np
import torch
import pandas as pd
from torchvision import tv_tensors
from sklearn.model_selection import train_test_split

class Fields(StrEnum):
    EVENT = 'event'
    X = 'x'
    Y = 'y'
    BEFORE = 'before'
    AFTER = 'after'
    LABEL = 'label'
    CHANGE = 'change'

class CutMixType(StrEnum):
    CIRCULAR = 'circular'
    RECTANGLE = 'rectangle'

class MultiTemporalDataset(Dataset, ABC):
    """
    Abstract base class for multi-temporal datasets.
    Subclasses should implement the `__len__` and `__getitem__` methods.
    """

    def __init__(self, path,
                 patch_size=256,
                 bands=list(range(13)),
                 n_prev_images=1,
                 transforms=None,
                 csv_path=None,
                 ignore_index=-1,
                 binary_change=False,
                 continous_labels=False,
                 cutmix=False,
                 cutmix_min_ratio=0.3,
                 cutmix_max_ratio=0.5,
                 cutmix_partial=False,
                 cutmix_type=CutMixType.RECTANGLE,
                 **kwargs):

        self.path = path
        self.transforms = transforms
        self.n_prev_images = n_prev_images
        self.bands = bands
        self.patch_size = patch_size
        self.ignore_index = ignore_index
        self.binary_change = binary_change
        self.continous_labels = continous_labels

        self.use_cutmix = cutmix
        self.cutmix_min_ratio = cutmix_min_ratio
        self.cutmix_max_ratio = cutmix_max_ratio
        self.cutmix_type = cutmix_type
        self.cutmix_partial = cutmix_partial

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.columns = [str(f) for f in Fields]

        self.data: pd.DataFrame = None

        if csv_path is None:
            self.data = self.load_data(path, n_prev_images,**kwargs)
        else:
            self.data = pd.read_csv(csv_path)
            self.data[Fields.BEFORE] = self.data[Fields.BEFORE].apply(ast.literal_eval)

        for c in self.columns:
            assert c in list(self.data.columns), f"Column '{c}' is missing in the dataset. Please check the data loading process."
        
        self.data[Fields.BEFORE] = self.data[Fields.BEFORE].apply(lambda x: x[-self.n_prev_images:])

        change_distribution = (self.data['change'].to_numpy().round(decimals=1) * 10).astype(int)
        self.index_by_change = {k: np.where(change_distribution == k)[0] for k in np.unique(change_distribution)}
        self.max_change_index = int(np.max(change_distribution))
        self.min_change_index = int(np.min(change_distribution))
    

    @abstractmethod
    def load_data(self, path, n_prev_images, **kwargs) -> pd.DataFrame:
        """Initialize the dataset.
        This method should be implemented by subclasses to set up the dataset.
        """
        pass
    
    def __len__(self):
        return len(self.data)


    @abstractmethod
    def load_sample(self, event, x, y, before, after, label, **kwargs):
        """Load a sample from the dataset.
        This method should be implemented by subclasses to load the actual data.

        Returns a dictionary with keys 'label', 'before', 'after'.
        """

        return dict(label=label, before=before, after=after)
    
    def apply_tranformations(self, sample, sample_data):
        if self.transforms is not None:
            if self.continous_labels:
                sample_data['label'] = [tv_tensors.Mask(l) for l in sample_data['label']]
            else:
                sample_data['label'] = tv_tensors.Mask(sample_data['label'])
            sample_data['after'] = tv_tensors.Image(sample_data['after'], dtype=torch.float32)
            sample_data['before'] = [tv_tensors.Image(b, dtype=torch.float32) for b in sample_data['before']]

            return self.transforms(sample_data)
        return sample_data

    
    def cutmix(self, sample, data):

        def generate_mask(height, width):

            def circular_mask(H, W):
                y = torch.arange(0, H).float()
                x = torch.arange(0, W).float()
                yy, xx = torch.meshgrid(y, x, indexing='ij')

                radius = min(H, W) // 2
                cy, cx = (H-1)/2, (W-1)/2
                mask = (((xx - cx)**2) + ((yy - cy)**2))**0.5 <= radius
                return mask.to(torch.uint8)

            mask = torch.zeros((self.patch_size, self.patch_size))
            w = random.randint(int(self.cutmix_min_ratio * width), int(self.cutmix_max_ratio * width))
            h = random.randint(int(self.cutmix_min_ratio * height), int(self.cutmix_max_ratio * height))
            
            start_x = random.randint(0, self.patch_size - h - 1) 
            start_y = random.randint(0, self.patch_size - w - 1)

            # generate mask wxh with gaussian distribution
            match self.cutmix_type:
                case CutMixType.CIRCULAR:
                    mask[start_x:start_x + h, start_y:start_y + w] = circular_mask(h, w)
                case CutMixType.RECTANGLE:
                    mask[start_x:start_x + h, start_y:start_y + w] = 1
            
            return mask
        
        change = round(sample['change'], 1)
        if change <= self.cutmix_min_ratio or change >= self.cutmix_max_ratio:
            change_index = round(sample['change'], 1)
            if  change_index >= 0.7:
                bin = self.index_by_change[self.min_change_index]
            else:
                bin = self.index_by_change[self.max_change_index]
            
            index2 = random.choice(bin)
            if index2 > len(self.data):
                print(index2, bin, len(self.data))
                print(self.index_by_change)
                assert False, "Index out of bounds in CutMix"
            sample2 = self.data.iloc[index2].to_dict()
            data2 = self.load_sample(**sample2)

            if self.continous_labels:
                empty_label = torch.zeros_like(data2['label'])
                empty_label[data2['label'] == self.ignore_index] = self.ignore_index
                data2['label'] = [empty_label.clone() for _ in range(self.n_prev_images-1)] + [data2['label']]


            l2 = data2[Fields.LABEL][-1]
            
            w_indices = torch.where(l2[0, :] == self.ignore_index)[0] 
            h_indices = torch.where(l2[:, 0] == self.ignore_index)[0] 
            w = w_indices[0].item() if len(w_indices) > 0 else self.patch_size
            h = h_indices[0].item() if len(h_indices) > 0 else self.patch_size

            mask = generate_mask(h, w)
            inv_mask = 1 - mask

            if self.cutmix_partial and change_index < 0.7:
                assert self.continous_labels, "Partial CutMix only supported for continuous labels"
                
                start_idx = random.randint(1, self.n_prev_images-1)
                for i in range(0, start_idx):
                    b1 = data[Fields.BEFORE][i]
                    b2 = data2[Fields.BEFORE][i]
                    data[Fields.BEFORE][i] = b1 * inv_mask + b2 * mask
                
                a2 = data2[Fields.AFTER]
                l2 = data2[Fields.LABEL][-1]
                for i in range(start_idx, self.n_prev_images):
                    b1 = data[Fields.BEFORE][i]
                    data[Fields.BEFORE][i] = b1 * inv_mask + a2 * mask
                    if i == start_idx: # first changed image
                        # labels are shifted by 1 - one less targets than images
                        data[Fields.LABEL][i-1] = data[Fields.LABEL][i-1] * inv_mask + l2 * mask
                    else:
                        data[Fields.LABEL][i-1] = data[Fields.LABEL][i-1] * inv_mask 
                
                data[Fields.AFTER] = data[Fields.AFTER] * inv_mask + a2 * mask
                data[Fields.LABEL][-1] = data[Fields.LABEL][-1] * inv_mask
                        
            else:
                for i in range(len(data[Fields.BEFORE])):
                    b1 = data[Fields.BEFORE][i]
                    b2 = data2[Fields.BEFORE][i]
                    data[Fields.BEFORE][i] = b1 * inv_mask + b2 * mask

                a1 = data[Fields.AFTER]
                a2 = data2[Fields.AFTER]
                data[Fields.AFTER] = a1 * inv_mask + a2 * mask

                if self.continous_labels:
                    data[Fields.LABEL][-1] = data[Fields.LABEL][-1] * inv_mask + data2[Fields.LABEL][-1] * mask
                else:
                    data[Fields.LABEL] = data[Fields.LABEL] * inv_mask + data2[Fields.LABEL] * mask
            
        return data
        
    
    def __getitem__(self, index):
        sample = self.data.iloc[index].to_dict()

        data = self.load_sample(**sample)
        
        if self.continous_labels:
            empty_label = torch.zeros_like(data['label'])
            empty_label[data['label'] == self.ignore_index] = self.ignore_index
            data['label'] = [empty_label.clone() for _ in range(self.n_prev_images-1)] + [data['label']]

        if self.use_cutmix:
            data = self.cutmix(sample, data)

        data = self.apply_tranformations(sample, data)

        sample.update(data)

        for k in list(sample.keys()):
            if k not in ['change', 'label', 'before', 'after']:
                del sample[k]

        assert (not self.continous_labels or isinstance(sample['label'], list) and
                not isinstance(sample['label'], list) or self.continous_labels), "Continous labels <=> label is of type list"

        if self.binary_change:
            sample['change'] = 1 if sample['change'] > 0 else 0

        
        sample['change'] = torch.tensor(sample['change'], dtype=torch.float32)


        return sample

    def to_csv(self, path):
        self.data.to_csv(path, index=False)
    
    def split_by_events(self, test_size=0.2, seed=42):
        events = self.data['event'].unique().tolist()

        train_scenes, test_scenes = train_test_split(events, test_size=test_size, random_state=seed)
        train = self.data[self.data['event'].isin(train_scenes)]
        test = self.data[self.data['event'].isin(test_scenes)]

        train_set = deepcopy(self)
        test_set = deepcopy(self)

        train_set.data = train
        test_set.data = test
        
        return train_set, test_set
    
    def random_split(self, test_size=0.2, seed=42):
        train, test = train_test_split(self.data, test_size=test_size, random_state=seed)
        train_set = deepcopy(self)
        test_set = deepcopy(self)


        train_set.data = train
        test_set.data = test

        chd_train = (train['change'].to_numpy().round(decimals=1) * 10).astype(int)
        train_set.index_by_change = {k: np.where(chd_train == k)[0] for k in np.unique(chd_train)}
        train_set.max_change_index = int(np.max(chd_train))
        train_set.min_change_index = int(np.min(chd_train))

        chd_test = (test['change'].to_numpy().round(decimals=1) * 10).astype(int)
        test_set.index_by_change = {k: np.where(chd_test == k)[0] for k in np.unique(chd_test)}
        test_set.max_change_index = int(np.max(chd_test))
        test_set.min_change_index = int(np.min(chd_test))

        return train_set, test_set
