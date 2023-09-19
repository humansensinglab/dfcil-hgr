from typing import  Optional, Sequence

import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functional as F 

try :
    from .helpers import one_hot
except :
    from helpers import one_hot


class Loss(nn.Module) :
    def __init__(self, 
        n_classes: int, 
        ignore: Optional[int] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> None :

        super().__init__()

        self.n_classes = n_classes
        if weights is None :           
            weights = [1.] * self.n_classes

        assert isinstance(weights, (list, tuple)), \
            f"weights must be of type (list, tuple)."            
        weights = np.array(weights)
        if ignore is not None :
            weights[ignore] = 0
        weights = torch.from_numpy(weights).to(torch.get_default_dtype())
        self.weights = weights.view(1, -1)

    
    def forward(self, 
        logits: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor :

        device = logits.device
        log_pred = F.log_softmax(logits, dim=1)
        target = one_hot(target, self.n_classes).to(device)

        view_shape = [1, -1] + [1] * (target.ndim-2)
        weights = self.weights.view(view_shape).to(device)
        loss = -weights * log_pred * target
        loss = loss.sum(1)
        if mask is not None :
            loss = loss * mask
            loss = loss.sum() / torch.count_nonzero(mask)
            return loss

        mult_ = (log_pred.shape[1] / np.prod(log_pred.shape))
        loss = loss.sum() * mult_
        return loss

    