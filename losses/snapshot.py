from typing import  Optional, Sequence

import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functional as F 


class Loss(nn.Module) :
    def __init__(self, 
        type_: str = 'mse',
    ) -> None :

        super().__init__();

        if type_ == 'mse' :
            self.loss_func = nn.MSELoss();
        elif type_ == 'l1' :
            self.loss_func = nn.L1Loss();
        else :
            raise NotImplementedError;

    
    def forward(self, 
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor :

        assert pred.shape == target.shape;
        assert pred.device == target.device;
        return self.loss_func(pred, target);


if __name__ == "__main__" :
    def gen_data(bs, c, h, w) :
        xs = torch.rand(bs, c, h, w);
        ys = torch.rand_like(xs);
        xs.requires_grad_(True);
        ys.requires_grad_(False);
        return xs, ys;

    bs, c, h, w = 4, 10, 1, 1;
    pred, label = gen_data(bs, c, h, w);
    loss = Loss('mse')(pred, label);
    loss.backward();
    print(loss);        