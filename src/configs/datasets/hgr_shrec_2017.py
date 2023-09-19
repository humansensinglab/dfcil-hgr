import os, os.path as osp

from copy import deepcopy
from easydict import EasyDict as edict

try :
    from .base import Config_Data as Base_Data
except :
    from base import Config_Data as Base_Data

class Config_Data(Base_Data) :
    """INFO: http://www-rech.telecom-lille.fr/shrec2017-hand/"""
    name = 'hgr_shrec_2017'

    raw_split_files = edict({
        'train': 'train_gestures.txt',
        'test': 'test_gestures.txt',
    })

    split_dir = 'split_files'
    split_types = (
        'single', # only single finger samples
        'multiple', # only multiple finger samples
        'agnostic', # single or multiple, don't care
        'specific', # separate class id for single and multiple
    )

    split_files = edict({
        'train': 'train.txt',
        'test': 'test.txt',
        'val': 'val.txt',
        'testval': 'val.txt',
    })


    __label_to_name = {
        0: 'grab',
        1: 'tap',
        2: 'expand',
        3: 'pinch',
        4: 'rotate_cw',
        5: 'rotate_ccw',
        6: 'swipe_right',
        7: 'swipe_left',
        8: 'swipe_up',
        9: 'swipe_down',
        10: 'swipe_x',
        11: 'swipe_+',
        12: 'swipe_v',
        13: 'shake',
    }

    label_to_name = {}

    __label_to_type = {
    	0: 'fine',
    	1: 'coarse',
    	2: 'fine',
    	3: 'fine',
    	4: 'fine',
    	5: 'fine',
    	6: 'coarse',
    	7: 'coarse',
    	8: 'coarse',
    	9: 'coarse',
    	10: 'coarse',
    	11: 'coarse',
    	12: 'coarse',
    	13: 'coarse',
    }    

    label_to_type = {}

    # class frequency - number of samples per class
    class_freq = {
        'train': {
            0: 72,
            1: 70,
            2: 67,
            3: 72,
            4: 73,
            5: 72,
            6: 73,
            7: 76,
            8: 71,
            9: 74,
            10: 68,
            11: 74,
            12: 67,
            13: 71,
            14: 74,
            15: 72,
            16: 68,
            17: 64,
            18: 69,
            19: 70,
            20: 67,
            21: 64,
            22: 72,
            23: 71,
            24: 72,
            25: 70,
            26: 62,
            27: 65,
        },

        'test': {
            0: 28,
            1: 30,
            2: 33,
            3: 28,
            4: 27,
            5: 28,
            6: 27,
            7: 24,
            8: 29,
            9: 26,
            10: 32,
            11: 26,
            12: 33,
            13: 29,
            14: 26,
            15: 28,
            16: 32,
            17: 36,
            18: 31,
            19: 30,
            20: 33,
            21: 36,
            22: 28,
            23: 29,
            24: 28,
            25: 30,
            26: 38,
            27: 35,
        },
    }


    def __init__(self, root_dir) :
        """ define all the data directories and subdirectories """
        assert osp.isdir(root_dir), f"Root directory {root_dir} not found."
        self.root_dir = os.path.expanduser(root_dir)

        for k in self.raw_split_files :
            self.raw_split_files[k] = osp.join(self.root_dir, self.raw_split_files[k])

        self.split_dir = osp.join(self.root_dir, self.split_dir)
        for k in self.split_types :
            os.makedirs(osp.join(self.split_dir, k), exist_ok=True)

        self._extend_label_maps_w_split_types()


    def assert_split_type(self, type_) :
        assert type_ in self.split_types, \
            f"Split type {type_} must be one of {self.split_types}"


    def get_split_filepath(self, type_, mode_) :
        self.assert_mode(mode_)
        self.assert_split_type(type_)
        fpath = osp.join(self.split_dir, type_, self.split_files[mode_])
        
        return fpath


    def _extend_label_maps_w_split_types(self) :
        self.label_to_name['single'] = deepcopy(self.__label_to_name)
        self.label_to_type['single'] = deepcopy(self.__label_to_type)
        self.label_to_name['multiple'] = deepcopy(self.__label_to_name)
        self.label_to_type['multiple'] = deepcopy(self.__label_to_type)
        self.label_to_name['agnostic'] = deepcopy(self.__label_to_name)
        self.label_to_type['agnostic'] = deepcopy(self.__label_to_type)

        self.label_to_name['specific'] = {}
        self.label_to_type['specific'] = {}        
        for k in self.__label_to_name :
            self.label_to_name['specific'][2*k] = self.__label_to_name[k] + '_1'        
            self.label_to_name['specific'][2*k+1] = self.__label_to_name[k] + '_2'
            self.label_to_type['specific'][2*k] = self.__label_to_type[k]        
            self.label_to_type['specific'][2*k+1] = self.__label_to_type[k]        


    def get_n_classes(self, type_) :
        return len(self.label_to_name[type_])

    
    def get_global_classes_per_task(self, tasks, index) :
        classes_per_task = tasks[:index + 1]
        classes_per_task = [item for sublist in classes_per_task for item in sublist]
        return classes_per_task


        
