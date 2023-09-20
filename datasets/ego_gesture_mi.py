import sys
import os, os.path as osp

from easydict import EasyDict as edict
from typing import Any, Callable, Sequence, Optional

import numpy as np
from torch.utils.data import Dataset as TorchDataset

try :
    from .base_mi import Dataset as BaseDataset
    from .helpers import *
    from . import transforms
except :
    from base_mi import Dataset as BaseDataset
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
        cfg: Config_Data,
        cfg_xforms: dict, 
        n_add_classes: int,
        n_known_classes: int = 0,
        rm_global_scale: bool = False,
        is_inverted: bool = False,
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
            is_inverted,
        );

        cfg_xforms = edict(cfg_xforms);
        self.xforms = transforms.get_transforms_from_cfg(cfg_xforms);
        self.to_tensor = transforms.ToTensor();


    def __get_orientation(self, pts) :
        assert pts.ndim in [2, 3];
        if pts.ndim == 3 :
            pts_left = pts[:, :21, :];
            pts_right = pts[:, 21:, :];
        else :
            pts_left = pts[:, :63];
            pts_right = pts[:, 63:];

        is_present_left = not np.allclose(pts_left, 0);
        is_present_right = not np.allclose(pts_right, 0);
        if is_present_left and is_present_right :
            orientation = 'b';
        elif is_present_left :
            orientation = 'l';
        else :
            orientation = 'r';
        
        return orientation;


    def __reset_pts_w_orientation(self, pts, orientation) :
        if orientation == 'b' :
            return pts;

        if orientation == 'l' :
            if pts.ndim == 3 :
                pts[:, 21:, :] = 0;
            else :
                pts[:, 63:] = 0;
        else :
            if pts.ndim == 3 :
                pts[:, :21, :] = 0;
            else :
                pts[:, :63] = 0;

        return pts;


    def __getitem_trainval(self, idx) :
        pts_path, label, size_seq, id_subject = self.file_list[idx];
        is_inverted = self.is_inverted[idx];

        data = self.read_pts(pts_path, is_inverted=is_inverted);
        if is_inverted :
            pts, label_feature, pred_feature = data;
        else :
            pts = data;

        orientation = self.__get_orientation(pts);

        pts = self.xforms(pts);
        pts = self.__reset_pts_w_orientation(pts, orientation);
        pts = self.to_tensor(pts);

        data = edict({
            'pts': pts,
            'label': label,
        });

        if is_inverted :
            data.label_feature = label_feature;
            data.pred_feature = pred_feature;

        return data;


    def __getitem_test(self, idx) :
        pts_path, label, size_seq, id_subject = self.file_list[idx];
        is_inverted = self.is_inverted[idx];

        data = self.read_pts(pts_path, is_inverted=is_inverted);
        if is_inverted :
            pts, label_feature, pred_feature = data;
        else :
            pts = data;

        orientation = self.__get_orientation(pts);
        pts = self.xforms(pts);
        pts = self.__reset_pts_w_orientation(pts, orientation);
        pts = self.to_tensor(pts);

        rel_path = pts_path[len(self.cfg.root_dir)+1:];
        rel_path = osp.dirname(rel_path).replace('/', '__');
        data = edict({
            'path': rel_path,
            'pts': pts,
            'label': label,
        });

        if is_inverted :
            data.label_feature = label_feature;    
            data.pred_feature = pred_feature;                

        return data;       


    def __getitem__(self, idx) :
        if (self.mode == 'train') \
            or (self.mode == 'val') :
            return self.__getitem_trainval(idx);
        elif self.mode.startswith('test') :
            return self.__getitem_test(idx);
        else :
            raise NotImplementedError;


if __name__ == "__main__" :

    test_loader = True;
    test_time = True;


    import yaml, random
    from pprint import pprint
    from tqdm import tqdm
    from configs.datasets import ego_gesture
    from utils.colors import *

    root_dir = '/data/datasets/agr/ego_gesture';
    # root_dir = '/data/ashubhra/agr/cvpr_2023/hgr_shrec_2017/vanilla/supcon/finetune/initial_1k/drop/class_6_11_4_10_2_8/inverted_samples';
    cfg_file = '../configs/params/oracle/ego_gesture/initial.yaml';
    split_type = 'agnostic';
    cfg_data = ego_gesture.Config_Data(root_dir);

    with open(cfg_file, 'rb') as f :
        cfg_params = edict(yaml.load(f, Loader=yaml.FullLoader));

    # mode = 'train';
    # mode = 'val';
    mode = 'test';
    dataset = Dataset(
                mode, 
                split_type,
                cfg_data,
                cfg_params.transforms[mode],
    );

    # dataset.merge_dataset(dataset);

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

    
