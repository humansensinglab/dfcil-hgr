import torch 
import torch.nn as nn
import torch.nn.functional as F 

def one_hot(x: torch.Tensor, n_classes: int) -> torch.Tensor :
    assert torch.is_tensor(x);
    shape = x.shape;
    oh = torch.zeros(
            (shape[0], n_classes) + shape[1:], 
            device=x.device, 
            dtype=x.dtype, 
    );

    return oh.scatter_(1, x.unsqueeze(1).long(), 1);