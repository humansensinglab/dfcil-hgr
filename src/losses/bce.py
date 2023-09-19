import torch
import torch.nn as nn
import torch.nn.functional as F 



class Loss(nn.Module) :
    def __init__(self, reduction) -> None :
    
        super().__init__()
        self.reduction = reduction
        self.ce_loss = nn.BCELoss(reduction=self.reduction)


    def forward(self, scores, target_scores):

        return self.ce_loss(torch.sigmoid(scores), target_scores) / len(scores)

     