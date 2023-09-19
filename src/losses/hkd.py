import torch.nn as nn


class Loss(nn.Module) :
    def __init__(self) -> None :
    
        super().__init__()
        self.hkd_loss = nn.L1Loss()


    def forward(self, scores, target_scores):

        return self.hkd_loss(scores, target_scores)

     