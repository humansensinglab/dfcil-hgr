import sys
import os.path as osp

from easydict import EasyDict as edict

import numpy as np

try :
    from .base import Dataset as BaseDataset
    from .helpers import *
    from . import transforms
except :
    from base import Dataset as BaseDataset
    from helpers import *
    import transforms    

sys.path.append('..')

from configs.datasets.base import Config_Data
from utils.stdio import *
from utils.misc import *



class Dataset(BaseDataset) :
    def __init__(self,
        mode: str,
        split_type: str,
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
        )

        cfg_xforms = edict(cfg_xforms)
        self.xforms = transforms.get_transforms_from_cfg(cfg_xforms)
        self.to_tensor = transforms.ToTensor()


    def __get_orientation(self, pts) :
        assert pts.ndim in [2, 3]
        if pts.ndim == 3 :
            pts_left = pts[:, :21, :]
            pts_right = pts[:, 21:, :]
        else :
            pts_left = pts[:, :63]
            pts_right = pts[:, 63:]

        is_present_left = not np.allclose(pts_left, 0)
        is_present_right = not np.allclose(pts_right, 0)
        if is_present_left and is_present_right :
            orientation = 'b'
        elif is_present_left :
            orientation = 'l'
        else :
            orientation = 'r'
        
        return orientation


    def __reset_pts_w_orientation(self, pts, orientation) :
        if orientation == 'b' :
            return pts

        if orientation == 'l' :
            if pts.ndim == 3 :
                pts[:, 21:, :] = 0
            else :
                pts[:, 63:] = 0
        else :
            if pts.ndim == 3 :
                pts[:, :21, :] = 0
            else :
                pts[:, :63] = 0

        return pts


    def __getitem_trainval(self, idx) :
        pts_path, label, size_seq, id_subject = self.file_list[idx]

        pts = self.read_pts(pts_path)
        # print('Just load = ', pts.shape)
        orientation = self.__get_orientation(pts)

        pts = self.xforms(pts)
        # print('After xforms = ', pts.shape)
        pts = self.__reset_pts_w_orientation(pts, orientation)
        # print('After reset = ', pts.shape)
        pts = self.to_tensor(pts)

        data = edict({
            'pts': pts,
            'label': label,
        })

        return data

    def __getitem_test(self, idx) :
        pts_path, label, size_seq, id_subject = self.file_list[idx]

        pts = self.read_pts(pts_path)
        orientation = self.__get_orientation(pts)
        pts = self.xforms(pts)
        pts = self.__reset_pts_w_orientation(pts, orientation)
        pts = self.to_tensor(pts)

        rel_path = pts_path[len(self.cfg.root_dir)+1:]
        rel_path = osp.dirname(rel_path).replace('/', '__')
        data = edict({
            'path': rel_path,
            'pts': pts,
            'label': label,
        })

        return data        


    def __getitem__(self, idx) :
        if (self.mode == 'train') \
            or (self.mode == 'val') :
            return self.__getitem_trainval(idx)
        elif self.mode.startswith('test') :
            return self.__getitem_test(idx)
        else :
            raise NotImplementedError

            

    
