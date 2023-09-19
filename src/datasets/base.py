import sys
import os.path as osp 

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from functools import partial

try :
    from .helpers import *
except :
    from helpers import *

sys.path.append('..')

from configs.datasets.base import Config_Data
from utils.stdio import *
from utils.misc import *


class Dataset(TorchDataset) :
    def __init__(self,
        mode: str,
        split_filepath: str,
        cfg: Config_Data,
        n_add_classes: int,
        n_known_classes: int,
        n_total_classes: int,
        rm_global_scale: bool,
        drop_seed: int,
    ) -> None :

        dataset_l = ['hgr_shrec_2017',  'ego_gesture']

        assert mode in cfg.modes, f"Training mode {mode} must be one of {cfg.modes}"
        assert cfg.name in dataset_l, f"Dataset name ({cfg.name}) must be one of {dataset_l}"
        assert osp.isdir(cfg.root_dir), "Root directory not found {cfg.root_dir}"

        self.mode = mode
        self.split_filepath = split_filepath
        self.cfg = cfg
        
        # classes to keep
        if n_add_classes > 0 :
            self.keep_class_l = get_add_class_list(
                                    n_add_classes, 
                                    n_known_classes, 
                                    n_total_classes, 
                                    drop_seed,
            )

        else :
            self.keep_class_l = None
            
        self.full_file_list, self.file_list = get_file_list(
                            self.cfg.name, 
                            self.cfg.root_dir, 
                            self.split_filepath, 
                            self.mode,
                            self.keep_class_l,
        )
        
        self.loader_pts = partial(
                            globals()['read_pts_' + cfg.name], 
                            rm_global_scale=rm_global_scale,
        )

        self.n_classes = None


    def __len__(self) :
        return len(self.file_list)


    def read_pts(self, fpath: str) -> np.ndarray :
        return self.loader_pts(fpath)

    
    def merge_dataset(self, other) :
        assert type(self) == type(other), \
            f"Type of this dataset {type(self)} does not match the other {type(other)}."
        
        print(f"(Before merging) Number of files = {len(self)}")
        print(f"(Before merging) Number of classes = {len(self.keep_class_l)}")

        self.file_list.extend(other.file_list)
        if self.keep_class_l is None :
            self.keep_class_l = other.keep_class_l
        else :
            self.keep_class_l = list( set(self.keep_class_l).union(other.keep_class_l) )
        if self.keep_class_l is not None :
            self.keep_class_l.sort()

        print(f"(After merging) Number of files = {len(self)}")
        print(f"(After merging) Number of classes = {len(self.keep_class_l)}")


    # naive coreset appending
    def append_coreset(self, coreset, ic, only=False):
        len_core = len(coreset)
        if (self.mode == 'train' or self.mode == 'val') and (len_core > 0):
            if only:
                self.file_list = coreset
            else:
                len_data = len(self.file_list)
                sample_ind = np.random.choice(len_core, len_data)
                if ic:
                    self.file_list.extend([coreset[i] for i in range(len(coreset))])
                else:
                    self.file_list.extend([coreset[i] for i in sample_ind])

    # naive coreset update
    def update_coreset(self, coreset, coreset_size, seen, class_mapping):
        state = np.random.get_state()
        np.random.seed(1994)
        mapped_targets = [class_mapping[str(self.full_file_list[i][1])] for i in range(len(self.full_file_list))]
        for k in seen:
            locs = (mapped_targets == k).nonzero()[0]
            num_data_k = max(1, round(coreset_size * len(locs) / 100))
            locs_chosen = np.random.permutation(locs)[:num_data_k]
            coreset.extend([self.full_file_list[loc] for loc in locs_chosen])
        np.random.set_state(state)
        return coreset


          