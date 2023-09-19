import torch.nn as nn
import torch.nn.functional as F



class Loss(nn.Module) :
    def __init__(self) -> None :
    
        super().__init__()


    def forward(self, scores, target_scores, weight=None, binary=False) :
        if binary :
            return F.binary_cross_entropy_with_logits(scores, target_scores, weight=weight)
        else :
            return F.cross_entropy(scores, target_scores, weight=weight)


     