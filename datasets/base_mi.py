import sys
import os, os.path as osp 

from typing import List, Tuple, Sequence, Any, Optional

import numpy as np
from torch.utils.data import Dataset as TorchDataset

from easydict import EasyDict as edict
from functools import partial

try :
    from .helpers import *
except :
    from helpers import *

sys.path.append('..');

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
        is_inverted: bool = False,
    ) -> None :

        dataset_l = ['hgr_shrec_2017',  'ego_gesture'];

        assert mode in cfg.modes, f"Training mode {mode} must be one of {cfg.modes}";
        assert cfg.name in dataset_l, f"Dataset name ({cfg.name}) must be one of {dataset_l}";
        assert osp.isdir(cfg.root_dir), "Root directory not found {cfg.root_dir}";

        self.mode = mode;
        self.split_filepath = split_filepath;
        self.cfg = cfg;

        # classes to keep
        self.keep_class_l = get_add_class_list(
                                n_add_classes, 
                                n_known_classes,
                                n_total_classes,
                                drop_seed,
        );
       
        self.file_list = get_file_list(
                            self.cfg.name, 
                            self.cfg.root_dir, 
                            self.split_filepath, 
                            self.mode,
                            self.keep_class_l,
        );

        self.is_inverted = [is_inverted] * len(self.file_list);
        
        self.loader_pts = partial(
                            globals()['read_pts_mi_' + cfg.name], 
                            rm_global_scale=rm_global_scale,
        );

        self.n_classes = None;


    def __len__(self) :
        return len(self.file_list);


    def read_pts(self, fpath: str, is_inverted: bool) -> np.ndarray :
        return self.loader_pts(fpath, is_inverted=is_inverted);


    def merge_dataset(self, other) :
        assert type(self) == type(other), \
            f"Type of this dataset {type(self)} does not match the other {type(other)}.";
        
        print(f"(Before merging) Number of files = {len(self)}");
        print(f"(Before merging) Number of classes = {len(self.keep_class_l)}");

        self.file_list.extend(other.file_list);
        self.is_inverted.extend(other.is_inverted);
        if self.keep_class_l is None :
            self.keep_class_l = other.keep_class_l;
        else :
            self.keep_class_l = list( set(self.keep_class_l).union(other.keep_class_l) );
        if self.keep_class_l is not None :
            self.keep_class_l.sort();

        print(f"(After merging) Number of files = {len(self)}");
        print(f"(After merging) Number of classes = {len(self.keep_class_l)}");







          