from easydict import EasyDict as edict

import torch

from .schedulers import *

def get_optimizer_scheduler(model, cfg) :
    cfg = edict(cfg)

    model_params = []
    if isinstance(model, (list, tuple)) :
        for m in model :
            model_params.append(list(m.parameters())) 
    else :
        model_params = model.parameters()

    if cfg.name == 'adam' :
        lr = cfg.lr if 'lr' in cfg else 0.001
        betas = cfg.betas if 'betas' in cfg else (0.9, 0.999)
        optimizer = torch.optim.Adam(
                        filter(lambda p: p.requires_grad, model_params), #model_params, 
                        lr=lr,
                        betas=betas,
        )

    elif cfg.name == 'sgd' :
        assert 'lr' in cfg
        lr = cfg.lr if 'lr' in cfg else 0.001
        momentum = cfg.momentum if 'momentum' in cfg else 0.9
        weight_decay = cfg.weight_decay if 'weight_decay' in cfg else 0.0001
        optimizer = torch.optim.SGD(
                        model_params, 
                        lr=lr,
                        momentum=momentum,
                        weight_decay=weight_decay,
        )        

    else :
        raise NotImplementedError

    scheduler = None    
    if 'scheduler' in cfg :
        cfg_sch = {
            'total_epochs': cfg.total_epochs,
            'n_steps_per_epoch': cfg.n_steps_per_epoch,
            **cfg.scheduler
        }

        scheduler = get_scheduler(optimizer, cfg_sch) 
    
    return optimizer, scheduler 


def optimizer_to_cuda(optimizer, rank) :
    for state in optimizer.state.values() :
        for k, v in state.items() :
            if torch.is_tensor(v) :
                state[k] = v.cuda(rank)