import torch.nn as nn
import torch.nn.functional as F 



class Loss(nn.Module) :
    def __init__(self) -> None :
    
        super().__init__()


    def forward(self, scores, target_scores, allowed_predictions, T=2.0, soft_t=False):
        """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].
        Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
        'Hyperparameter': temperature"""


        log_scores_norm = F.log_softmax(scores[:, allowed_predictions] / T, dim=1)
        if soft_t:
            targets_norm = target_scores
        else:
            targets_norm = F.softmax(target_scores[:, allowed_predictions] / T, dim=1)

        # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
        KD_loss_unnorm = -(targets_norm * log_scores_norm)
        KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)                      #--> sum over classes
        KD_loss_unnorm = KD_loss_unnorm.mean()                          #--> average over batch

        # normalize
        KD_loss = KD_loss_unnorm # * T**2

        return KD_loss

     