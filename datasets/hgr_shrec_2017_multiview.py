import sys
import os, os.path as osp

from easydict import EasyDict as edict
from typing import List, Any, Callable, Sequence, Optional

import numpy as np
from torch.utils.data import Dataset as TorchDataset

try :
    from .base import Dataset as BaseDataset
    from .helpers import *
    from . import transforms
except :
    from base import Dataset as BaseDataset
    from helpers import *
    import transforms    

sys.path.append('..');

from configs.datasets.base import Config_Data
from utils.stdio import *
from utils.misc import *



class Dataset(BaseDataset) :
    def __init__(self,
        mode: str,
        split_type: str,
        n_views: int,
        cfg: Config_Data,
        cfg_xforms: dict, 
        n_add_classes: int = -1,
        n_known_classes: int = 0, 
        rm_global_scale: bool = False,
        drop_seed: int = -1,
    ) -> None :

        super().__init__(
            mode, 
            cfg.get_split_filepath(split_type, mode), 
            cfg, 
            n_add_classes, 
            n_known_classes, 
            cfg.get_n_classes(split_type),
            rm_global_scale,
            drop_seed,
        );

        cfg_xforms = edict(cfg_xforms);
        self.xforms = transforms.get_transforms_from_cfg(cfg_xforms);
        self.to_tensor = transforms.ToTensor();

        self.n_views = n_views;

    def __getitem_train(self, idx) :
        pts_path, label, size_seq, id_subject = self.file_list[idx];

        pts = self.read_pts(pts_path);
        pts_l = [];
        for _ in range(self.n_views) :
            pts_l.append( self.to_tensor(self.xforms(pts)) );

        pts = torch.stack(pts_l, dim=0);

        data = edict({
            'pts': pts,
            'label': label,
        });

        return data;


    def __getitem__(self, idx) :
        if (self.mode == 'train') :
            return self.__getitem_train(idx);
        else :
            raise NotImplementedError;

    

if __name__ == "__main__" :

    test_loader = True;
    test_time = False;


    import yaml, random
    from pprint import pprint
    from tqdm import tqdm
    from configs.datasets import hgr_shrec_2017
    from utils.colors import *

    root_dir = '/data/datasets/agr/shrec2017';
    cfg_file = '../configs/params/oracle/initial.yaml';
    split_type = 'specific';
    n_views = 2;
    cfg_data = hgr_shrec_2017.Config_Data(root_dir);

    with open(cfg_file, 'rb') as f :
        cfg_params = edict(yaml.load(f, Loader=yaml.FullLoader));

    # mode = 'train';
    mode = 'train';
    dataset = Dataset(
                mode, 
                split_type,
                n_views,
                cfg_data,
                cfg_params.transforms[mode],
    );

    if not test_loader :

        # # test label maps 
        # pprint(cfg_data.label_map_arr.tolist());
        # print(np.unique(cfg_data.label_map_arr));

        print("Number of samples = ", len(dataset));
        idx = random.randint(0, len(dataset)-1);
        data = dataset[idx];
        pts = data.pts;
        label = data.label;

        print('pts', pts.dtype, pts.shape, pts.max(), pts.min());
        print('label', label);

        sys.exit();

    # test loader
    from torch.utils.data import DataLoader as TorchDataLoader

    dataloader = TorchDataLoader(dataset, batch_size=4, shuffle=True);

    if not test_time :
        iter_loader = iter(dataloader);
        data = next(iter_loader);
        print(type(data));

        pts = data.pts;
        label = data.label;

        print('pts', pts.dtype, pts.shape, pts.max(), pts.min());
        print('label', label);

        sys.exit(); 

    for data in tqdm(dataloader) :
        pts = data.pts;
        label = data.label;
        pts = pts.cuda(0, non_blocking=True);
        label = label.to(0, non_blocking=True);

    sys.exit();               

    
