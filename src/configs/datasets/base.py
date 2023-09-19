from easydict import EasyDict as edict


class Config_Data(object) :
    name = None

    modes = ['train', 'val', 'test', 'testval', 'testtrain']
    split_dir = 'split_files'
    split_files = edict({
        'train': 'train.txt',
        'val': 'val.txt',
        'test': 'test.txt',
        'testval': 'test.txt',
        'testtrain': 'train.txt',
    })

    label_to_name = {}

    # class frequency
    class_freq = { }  

    def assert_mode(self, mode_) :
        assert mode_ in self.modes, \
            f"Split mode {mode_} must be one of {self.modes}"        

    def get_n_classes(self) :
        raise NotImplementedError

    def get_split_filepath(self) :
        raise NotImplementedError