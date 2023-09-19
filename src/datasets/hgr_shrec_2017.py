import sys
import os.path as osp

from easydict import EasyDict as edict

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

    def __getitem_trainval(self, idx) :
        pts_path, label, size_seq, id_subject = self.file_list[idx]

        pts = self.read_pts(pts_path)
        pts = self.xforms(pts)
        pts = self.to_tensor(pts)

        data = edict({
            'pts': pts,
            'label': label,
        })

        return data

    def __getitem_test(self, idx) :
        pts_path, label, size_seq, id_subject = self.file_list[idx]

        pts = self.read_pts(pts_path)
        pts = self.xforms(pts)
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

            

    
