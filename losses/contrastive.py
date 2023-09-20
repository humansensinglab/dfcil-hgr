"""Modified from source: https://github.com/HobbitLong/SupContrast/blob/master/losses.py """

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor

from typing import Optional, Sequence

def _assert_isnan(x: Tensor) : 
    assert not torch.any(torch.isnan(x));


class Loss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, 
        n_views: int,
        is_supervised: bool,
        temperature: float = 0.07, 
    ) -> None :

        super().__init__();

        assert is_supervised, \
            "Implementation not checked for unsupervised computations.";

        self.n_views = n_views;
        self.temperature = temperature;
        self.is_supervised = is_supervised;


    def forward(self, 
        features: Tensor, 
        labels: Optional[Tensor] = None, 
        mask_old: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,        
    ) -> Tensor :

        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device;
        dtype = features.dtype;

        assert len(features.shape) >= 3, \
            f"features shape needs to be [batch_size, n_views, ...], got {features.shape}";
        assert features.size(1) == self.n_views, \
            f"Dimension along axis 1 must equal n_views {self.n_views}, got {features.size(1)}";

        bs, nv = features.shape[:2]; 
        features = features.view(bs, nv, -1);

        if (labels is None and mask is None) or \
                (labels is not None and mask is not None ):

            raise ValueError("One of `labels` and `mask` must be provided, not both.");
        elif labels is not None:
            labels = labels.view(-1, 1);
            assert labels.numel() == bs, \
                f"#Elements in labels must equal batch size {bs}, got {labels.numel()}";
            mask = torch.eq(labels, labels.T).to(dtype).to(device);
        else:
            mask = mask.to(dtype).to(device);

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0); # bs * nv x d
        if self.is_supervised :
            anchor_feature = contrast_feature;
            anchor_count = self.n_views;
        else :
            anchor_feature = features[:, 0];
            anchor_count = 1;


        # compute logits
        anchor_dot_contrast = (anchor_feature @ contrast_feature.T) / self.temperature; # (bs * nv) x (bs * nv)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True);
        logits = anchor_dot_contrast - logits_max; #logits_max.detach();

        # tile mask
        mask = mask.repeat(anchor_count, self.n_views);
        # mask-out self-contrast cases
        logits_mask = (1 - torch.eye(mask.size(0))).to(device);  
        mask = mask * logits_mask;

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask;
        exp_logits = torch.clamp(exp_logits, min=1e-3);
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True));
        # compute mean of log-likelihood over positive
        if mask_old is not None :
            log_prob = log_prob[mask_old];
            mask = mask[mask_old];
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1);
        loss = - mean_log_prob_pos.mean();
        _assert_isnan(loss);
        return loss;


if __name__ == "__main__" :
    def gen_data(bs, nv, d, c) :
        xs = torch.rand(bs, nv, d);
        ys = torch.randint(0, c, (bs, ));      
        xs.requires_grad_(True);
        return xs, ys;

    bs, nv, d, c = 16, 2, 16, 10;
    logits, label = gen_data(bs, nv, d, c);
    print(label)
    loss = Loss(nv, True)(logits, label);
    loss.backward();
    print(loss);    