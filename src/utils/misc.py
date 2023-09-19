import os, os.path as osp
import shutil
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import json

from easydict import EasyDict as edict
import operator

import random
import numpy as np 
import torch

from typing import List
from contextlib import redirect_stdout

def is_np_bool(x: np.ndarray) :
    return x.dtype.type == np.bool_

def init_best_measure_info(tag_: str, type_: str) :
    type_l = ['accuracy', 'error']
    assert type_ in type_l, f"type_ {type_} must be one of {type_l}"
    info = edict()
    info.tag = tag_
    info.type = type_
    
    if info.type == 'accuracy' :
        info.val = 0.
        info.func = operator.gt # x > bm
    else :
        info.val = 1e6
        info.func = operator.lt # x < bm
    
    return info


def set_seed(seed: int=3407) :
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True   
    print(f"Setting deterministic")  


def seed_worker(worker_id):
    worker_seed = 3407
    np.random.seed(worker_seed)
    random.seed(worker_seed)      


def load_state_dict_single(state_dict, model, optimizer=None, scheduler=None) :
    """Load a single instance of model, optimizer, scheduler state dicts."""

    model.load_state_dict(state_dict['model'])    
    if (optimizer is None) and (scheduler is None) :
        print("Optimizer and scheduler are not provided, so will NOT be loaded.")
        return

    optimizer.load_state_dict(state_dict['optimizer'])
    scheduler.load_state_dict(state_dict['scheduler'])


def get_state_dict_single(model, optimizer, scheduler, is_distributed) :
    """Get state dicts for a single instance of model, 
    optimizer (optional), and scheduler (optional)."""
    return {
        'model': model.module.state_dict() if is_distributed else model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None,
    }


def get_checkpoint_dir(log_dir, checkpoint_subdir='checkpoints') :
    checkpoint_dir = osp.join(log_dir, checkpoint_subdir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_checkpoint_path(
        log_dir, epoch=None, 
        save_best_only=None, save_last_only=None, ) :

    if save_best_only or save_last_only : 
        return osp.join(get_checkpoint_dir(log_dir), 'checkpoint_last.pth.tar')
    assert epoch is not None, f"epoch is required to save the checkpoint."
    return osp.join(get_checkpoint_dir(log_dir), 'checkpoint_' + str(epoch) + '.pth.tar')


def get_last_checkpoint_path( log_dir, max_epoch=1000 ) :
    ckpt_path = get_checkpoint_path(log_dir, save_last_only=True)
    if osp.isfile(ckpt_path) :
        return ckpt_path
    
    for epoch in range(max_epoch, -1, -1) :
        ckpt_path = get_checkpoint_path(log_dir, epoch)
        if osp.isfile(ckpt_path) :
            return ckpt_path

    return None


def get_best_model_path(log_dir) :
    return osp.join(get_checkpoint_dir(log_dir), 'model_best.pth.tar') 


def save_checkpoint(log_dir, states, epoch, is_best=None,
                    save_best_only=None, save_last_only=None) :

    checkpoint_path = get_checkpoint_path(log_dir, epoch, 
                            save_best_only, save_last_only)
    if save_last_only :
        if osp.isfile(checkpoint_path) :
            os.remove(checkpoint_path)

    torch.save(states, checkpoint_path)

    if is_best :
        best_model_path = get_best_model_path(log_dir)
        shutil.copyfile(checkpoint_path, best_model_path)


def tensor_dict_to_cuda(tdict: dict, gpu: int) :
    for k in tdict :
        if isinstance(tdict[k], dict) :
            tensor_dict_to_cuda(tdict[k], gpu)
            continue
        
        if torch.is_tensor(tdict[k]) :
            tdict[k] = tdict[k].cuda(gpu, non_blocking=True)    


def print_array_stats(arr) :
    assert isinstance(arr, np.ndarray) or isinstance(arr, torch.Tensor)

    if isinstance(arr, np.ndarray) :
        device = 'cpu'
    else :
        device = arr.device if arr.is_cuda else 'cpu'

    dtype = arr.dtype
    shape_ = arr.shape
    min_, max_ = arr.min(), arr.max()
    mean_, std_ = arr.mean(), arr.std()
    print("(Device, Dtype, Shape, Min, Max, Mean, Std) = ", end='')
    print(f"({device}, {dtype}, {shape_}, {min_:.4f}, {max_:.4f}, {mean_:.4f}, {std_:.4f})")

    iou_str = ' & '.join(iou_str)
    print(iou_str)
    

def print_ious(iou_all, n_classes, label_to_name, save_to_file, file_path) :
    print('*' * 80)
    print(f"IoU (mean) = {iou_all.iou:.2f} %")
    print('-' * 80)

    iou_str = [f"{iou_all.iou:.1f}"]
    for i in range(n_classes) :
        name = label_to_name[i]
        print(f"({i}) {name.capitalize()} = {iou_all[name]:.2f} %") 
        iou_str.append(f"{iou_all[name]:.1f}")
        
    print('*' * 80)  

    iou_str = ' & '.join(iou_str)
    print(iou_str)

    if save_to_file:
        with open(file_path, 'w+') as f:
            with redirect_stdout(f):
                print('*' * 80)
                print(f"IoU (mean) = {iou_all.iou:.2f} %")
                print('-' * 80)

                iou_str = [f"{iou_all.iou:.1f}"]
                for i in range(n_classes) :
                    name = label_to_name[i]
                    print(f"({i}) {name.capitalize()} = {iou_all[name]:.2f} %") 
                    iou_str.append(f"{iou_all[name]:.1f}")
                    
                print('*' * 80)  

                iou_str = ' & '.join(iou_str)
                print(iou_str)


def invert_list(arr: List[int], max_val: int) -> List[int] :
    arr_full = set(range(max_val))
    return list(arr_full.difference(arr))


def get_add_class_list(
    n_add_classes: int, 
    n_known_classes: int,
    n_total_classes: int,
    drop_seed: int,
) -> List[int] :

    if drop_seed < 0 :
        return [x for x in range(n_known_classes, n_known_classes + n_add_classes)]    

    class_l = np.random.RandomState(seed=drop_seed).permutation(n_total_classes)
    class_l = class_l[n_known_classes : n_known_classes + n_add_classes].tolist()
    #class_l.sort()
    return class_l


def summarize_results(root_log_dir = None, n_trials = 1, max_task=1) :
    metrics_dict = {}
    metrics_dict['local'] = {'raw': []}
    metrics_dict['global'] = {'raw': []}
    metrics_dict['old'] = {'raw': []}
    metrics_dict['new'] = {'raw': []}
    for trial in range(1, n_trials+1) :
        trial_log_dir = osp.join(root_log_dir, f'trial_{trial}')
        # Load metrics
        for test_mode in ['local', 'global', 'old', 'new'] :
            if max_task == 1:
                metrics_dir = osp.join(trial_log_dir, 'task_0', 'test', f"test_metrics_{test_mode}.json")
            else:
                metrics_dir = osp.join(trial_log_dir, f"test_metrics_{test_mode}.json")
            with open(metrics_dir, 'r') as f:
                metrics = json.load(f)
            if max_task == 1:
                metrics_dict[test_mode]['raw'].append(metrics['Acc'])
            else:
                metrics_dict[test_mode]['raw'].append(metrics['acc_metrics'])
    # Compute mean and std
    for test_mode in ['local', 'global', 'old', 'new'] :
        metrics_dict[test_mode]['raw'] = np.array(metrics_dict[test_mode]['raw'])
        metrics_dict[test_mode]['mean'] = np.mean(metrics_dict[test_mode]['raw'], axis=0)
        metrics_dict[test_mode]['std'] = np.std(metrics_dict[test_mode]['raw'], axis=0)
        metrics_dict[test_mode]['raw'] = metrics_dict[test_mode]['raw'].tolist()
        metrics_dict[test_mode]['mean'] = metrics_dict[test_mode]['mean'].tolist()
        metrics_dict[test_mode]['std'] = metrics_dict[test_mode]['std'].tolist()
    # Save metrics in json files
    with open(osp.join(root_log_dir, f"test_metrics.json"), 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    # Save matplotlib figure
    if max_task != 1:
        n_tasks = len(metrics_dict['local']['mean'])
        color_l = [(0.5, 0.5, 0.9), (0.9, 0.5, 0.5), (0.5, 0.9, 0.5), (0.9, 0.9, 0.5)]
        fig, ax = plt.subplots()
        x_index = range(1, n_tasks+1)
        ax.errorbar(x_index, metrics_dict['local']['mean'], yerr=metrics_dict['local']['std'], 
                    fmt ='-o', label='local', color=color_l[0])
        ax.errorbar(x_index, metrics_dict['global']['mean'], yerr=metrics_dict['global']['std'], 
                    fmt ='-o', label='global', color=color_l[1])
        ax.errorbar(x_index, metrics_dict['old']['mean'], yerr=metrics_dict['old']['std'],
                    fmt ='-o', label='old', color=color_l[2])
        ax.errorbar(x_index, metrics_dict['new']['mean'], yerr=metrics_dict['new']['std'],
                    fmt ='-o', label='new', color=color_l[3])
        ax.set_xlabel('Task')
        ax.set_ylabel('% Accuracy')
        ax.set_title('Accuracy per task')
        ax.legend()
        plt.ylim([0, 105])
        plt.xticks(x_index)
        fig.savefig(osp.join(root_log_dir, f"test_metrics.png"))
        plt.close(fig)