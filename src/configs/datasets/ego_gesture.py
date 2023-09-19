import os, os.path as osp

from copy import deepcopy
from easydict import EasyDict as edict

try :
    from .base import Config_Data as Base_Data
except :
    from base import Config_Data as Base_Data

class Config_Data(Base_Data) :
    """INFO: http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html"""
    name = 'ego_gesture'

    n_classes = 83
    n_subjects = 50
    subject_splits = edict()
    subject_splits.test = (2, 9, 11, 14, 18, 19, 28, 31, 41, 47)
    subject_splits.val = (1, 7, 12, 13, 24, 29, 33, 34, 35, 37)
    subject_splits.train = tuple(set(range(1, n_subjects+1)).difference(
                                subject_splits.val + subject_splits.test
    ))


    split_dir = 'split_files'
    split_files = edict({
        'train': 'train.txt',
        'val': 'val.txt',
        'testval': 'val.txt',
        'testtest': 'test.txt',
        'testtrain': 'train.txt',
        'test': 'test.txt',
    })


    __label_to_name = {i: str(i).zfill(2) for i in range(n_classes)}
    label_to_name = {}

    # class frequency - number of samples per class
    class_freq = {
        'train': { },
        'val': { },
        'test': { },
    }


    def __init__(self, root_dir) :
        """ define all the data directories and subdirectories """
        assert osp.isdir(root_dir), f"Root directory {root_dir} not found."
        self.root_dir = os.path.expanduser(root_dir)

        self.split_dir = osp.join(self.root_dir, self.split_dir)
        os.makedirs(self.split_dir, exist_ok=True)

        self._extend_label_maps_w_split_types()


    def assert_split_type(self, type_) :
        pass


    def get_split_filepath(self, type_, mode_) :
        self.assert_mode(mode_)
        self.assert_split_type(type_)
        fpath = osp.join(self.split_dir, self.split_files[mode_])
        return fpath


    def get_n_classes(self, type_) :
        return self.n_classes


    def get_global_classes_per_task(self, tasks, index) :
        classes_per_task = tasks[:index + 1]
        classes_per_task = [item for sublist in classes_per_task for item in sublist]
        return classes_per_task

    
    def _extend_label_maps_w_split_types(self) :
        self.label_to_name['single'] = deepcopy(self.__label_to_name)
        self.label_to_name['multiple'] = deepcopy(self.__label_to_name)
        self.label_to_name['agnostic'] = deepcopy(self.__label_to_name)

        self.label_to_name['specific'] = {}
        for k in self.__label_to_name :
            self.label_to_name['specific'][2*k] = self.__label_to_name[k] + '_1'        
            self.label_to_name['specific'][2*k+1] = self.__label_to_name[k] + '_2'



        
