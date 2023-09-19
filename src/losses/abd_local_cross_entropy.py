import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module) :
    def __init__(self) -> None :
    
        super().__init__()


    def forward(self, scores, target_scores, data_weights=None, binary=False) :
        if binary :
            loss = (F.binary_cross_entropy_with_logits(scores, target_scores, reduction='none')* data_weights).mean()
        else :
            loss = (F.cross_entropy(scores, target_scores.long(), reduction='none') * data_weights).mean()
        return loss


     