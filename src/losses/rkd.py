# -*- coding: utf-8 -*-

"""
Pytorch port of Relation Knowledge Distillation Losses.

credits:
    https://github.com/lenscloth/RKD/blob/master/metric/utils.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(
        min=eps
    )

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


# class RKDAngleLoss(nn.Module):
class Loss(nn.Module):
    def __init__(
        self,
        in_dim1: int = 0,
        in_dim2: int = None,
        proj_dim: int = None,
    ):
        super().__init__()

        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim1 if in_dim2 is None else in_dim2

        if proj_dim is None:
            proj_dim = min(self.in_dim1, self.in_dim2)

        self.proj_dim = proj_dim

        self.embed1 = self.embed2 = nn.Identity()
        if in_dim1 > 0:
            self.embed1 = nn.Linear(self.in_dim1, self.proj_dim)
            self.embed2 = nn.Linear(self.in_dim2, self.proj_dim)

    def forward(self, student, teacher):
        # N x C
        # N x N x C
        student, teacher = self.embed1(student), self.embed2(teacher)

        td = teacher.unsqueeze(0) - teacher.unsqueeze(1)
        norm_td = F.normalize(td, p=2, dim=2)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = student.unsqueeze(0) - student.unsqueeze(1)
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.l1_loss(s_angle, t_angle)
        return loss


class RKDDistanceLoss(nn.Module):
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d)
        return loss
