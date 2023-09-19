from easydict import EasyDict as edict

import torch

def get_scheduler(optimizer, cfg) :
    cfg = edict(cfg)

    if cfg.name == 'linear' :
        assert 'start_factor' in cfg
        assert 'end_factor' in cfg
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=cfg.start_factor,
            end_factor=cfg.end_factor,
            total_iters=cfg.total_epochs,
        )      
    
    else :
        raise NotImplementedError


def step_scheduler(scheduler) :
    if isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR) \
            or isinstance(scheduler, torch.optim.lr_scheduler.LinearLR) :
        scheduler.step()
    else :
        raise NotImplementedError


def scheduler_to_cuda(scheduler, rank) :
    if not hasattr(scheduler, 'state') :
        return

    for state in scheduler.state.values() :
        for k, v in state.items() :
            if torch.is_tensor(v) :
                state[k] = v.cuda(rank)           
