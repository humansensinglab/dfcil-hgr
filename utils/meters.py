import torch
import math
import numpy as np

from typing import Sequence, Optional

from easydict import EasyDict as edict

@torch.no_grad()
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
class PIoUMeter(object):
    def __init__(self, 
        n_classes: int, 
        label_to_name: dict, 
    ) -> None :

        self.eps = 1e-6;
        self.n_classes = n_classes;
        self.label_to_name = label_to_name;

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu';
        self.reset();

    def reset(self):
        self.name_to_label = {n: l for l, n in self.label_to_name.items()};
        # row - pred, col-target
        self.conf_matrix = torch.zeros(
            (self.n_classes, self.n_classes), device=self.device).long();


    def update(self, output, target, n=1, take_max=True) :
        if take_max :
            _, output = output.max(1);        
        output = output.long().view(-1);
        target = target.long().view(-1);

        assert (output.shape==target.shape);
        
        ids = torch.stack([output, target], dim=0);
        values = torch.ones((ids.shape[-1]), device=self.device).long();

        self.conf_matrix.index_put_(tuple(ids), values, accumulate=True);


    def stats(self) :
        conf = self.conf_matrix.clone().float();

        tp = conf.diag();
        fp = conf.sum(1) - tp;
        fn = conf.sum(0) - tp;

        intersection = tp;
        union = tp + fp + fn;
        present = union.gt(0);

        ious = intersection / (union + 1e-6);
        iou_mean = ious[present].mean().item() * 100;
        ious = (ious.data.cpu().numpy() * 100).tolist();

        iou_dict = edict({'iou': round(iou_mean, 2)});
        for i in range(len(ious)) :
            name = self.label_to_name[i];
            iou_ = ious[i];
            iou_dict[name] = round(iou_, 2);
        
        return iou_dict;
        

    def accuracies(self, class_ids=None) :
        conf = self.conf_matrix.clone().float();

        if class_ids is None :
            total = conf.sum().item();
            correct = conf.diag().sum().item();

        else :
            total = conf[:, class_ids].sum().item();
            correct = conf.diag()[class_ids].sum().item();

        acc = (correct / total) * 100;
        return acc;    
