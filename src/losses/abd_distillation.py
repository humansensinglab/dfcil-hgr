import torch.nn as nn


class Loss(nn.Module) :
    def __init__(self) -> None :

        super().__init__()
        self.kd_criterion = nn.MSELoss(reduction="none")
        self.mu =1 


    def forward(self, scores, target_scores, dw_KD = 1):

        return self.mu * (self.kd_criterion(scores, target_scores).sum(dim=1) * dw_KD).mean() / (scores.size(1))