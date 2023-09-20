import os, os.path as osp
import shutil

import numpy as np

from torch.utils.tensorboard import SummaryWriter

class TensorBoardLogger(object) :
    def __init__(self, log_dir) :
        if osp.isdir(log_dir) : shutil.rmtree(log_dir);
        self.log_dir = log_dir;
        self.writer = SummaryWriter(log_dir);

    def update(self, stats, step, prefix='') :
        """update tensorboard with multi-modal stats"""
        for tag, value in stats.items() :
            self.scalar_summary("{}/{}".format(prefix, tag), value, step);

    def flush(self) :
        self.writer.flush();

    def scalar_summary(self, tag, value, step) :
        self.writer.add_scalar(tag, value, step);

    def image_summary(self, tag, value, step) :
        self.writer.add_images(tag, value, step);       

    def close(self) :
        self.writer.close();