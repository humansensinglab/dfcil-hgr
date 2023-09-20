# new-old separate loss + low new-old tension with detachment.
import sys
import os, os.path as osp

import argparse
import random
import shutil
import time

import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp

import yaml
import importlib
from pprint import pprint
from easydict import EasyDict as edict
from tqdm import tqdm

from copy import deepcopy

SUB_DIR_LEVEL = 3; # level of this subdirectory w.r.t. root of the code
sys.path.append(osp.join(*(['..'] * SUB_DIR_LEVEL)));

import model_defs
import optimizers as optimizer_defs
import losses as loss_defs
import utils

parser = argparse.ArgumentParser(description='Supervised Contrastive Pretraining.')
parser.add_argument('--dataset', type=str, required=True, help='name of the dataset.');
parser.add_argument('--split_type', type=str, required=True, help='type of data split (if applicable).');
parser.add_argument('--cfg_file', type=str, required=True, help='config file to load experimental parameters.');
parser.add_argument('--root_dir', type=str, required=True, help='root directory containing the dataset.');
parser.add_argument('--log_dir', type=str, required=True, help='directory for logging.');
parser.add_argument('--pretrain_dir', type=str, required=True, 
                    help='directory used for pretraining logs.');
parser.add_argument('--inverted_data_dir', type=str, required=True, 
                    help='directory containing inverted data.');                    

parser.add_argument('--n_add_classes', type=int, required=True, 
                    help='#classes to add (-1) to keep all.');
parser.add_argument('--n_known_classes', type=int, required=True, 
                    help='#classes known before.');
parser.add_argument('--drop_seed', type=int, required=True, 
                    help='seed used for class dropping.');                    

parser.add_argument('--proto_fname', type=str, default='proto.pkl', 
                    help='name of the proto file.');
parser.add_argument('--clf_fname', type=str, default='classifier.pkl', 
                    help='name of the proto file.');                    
parser.add_argument('--use_reduced', action='store_true', 
                    help='whether to use the reduced representation.');

parser.add_argument('--save_last_only', action='store_true', 
                    help='whether to save the last epoch only.');
parser.add_argument('--save_epoch_freq', type=int, default=1,
                    help='epoch frequency to save checkpoints.');

best_measure_info = utils.init_best_measure_info('iou', 'accuracy');


def main() :
    args = parser.parse_args();
    args.dist_url = 'tcp://127.0.0.1:' + utils.get_free_port();
    utils.print_argparser_args(args);

    utils.set_seed();
   
    n_gpus = torch.cuda.device_count();
    assert n_gpus>0, "A GPU is required for execution.";

    if n_gpus > 1 :
        mp.spawn(main_worker, nprocs=n_gpus, 
                args=(n_gpus, args),
        );

    else :
        main_worker(0, 1, args);


def main_worker(gpu, n_gpus, args) :
    global best_measure_info;

    args.gpu = gpu;

    with open(args.cfg_file, 'rb') as f :
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader));
    # pprint(cfg); sys.exit();


    cfg_data = getattr(importlib.import_module(
                    '.' + args.dataset, 
                    package='configs.datasets'),
                'Config_Data')(args.root_dir);    

    assert cfg.batch_size.mi % n_gpus == 0, \
            f"Batch size {cfg.batch_size.mi} must be divisible by #GPUs {n_gpus}";
    assert cfg.batch_size.new % n_gpus == 0, \
            f"Batch size {cfg.batch_size.new} must be divisible by #GPUs {n_gpus}";            
    assert cfg.workers % n_gpus == 0, \
            f"Workers {cfg.workers} must be divisible by #GPUs {n_gpus}";

    is_distributed = n_gpus > 1;

    # split batch size and workers across gpus
    for k in cfg.batch_size :
        cfg.batch_size[k] = cfg.batch_size[k] // n_gpus;
    cfg.workers = cfg.workers // n_gpus;

    print(f"Using GPU = {args.gpu} with (batch_size, workers) = ({cfg.batch_size}, {cfg.workers})");
    torch.cuda.set_device(args.gpu);

    model = model_defs.get_model(edict({
            'n_classes': cfg_data.get_n_classes(args.split_type),
            **cfg.model
    }) );
    model_defs.print_n_params(model);
    # print(model);

    if is_distributed :
        dist.init_process_group(
                backend='nccl', 
                init_method=args.dist_url,
                world_size=n_gpus,
                rank=args.gpu,
        );    

    new_train_dataset = \
        getattr(
            importlib.import_module(
                '.' + args.dataset + '_multiview', package='datasets'), 
            'Dataset')(
                'train',
                args.split_type,
                cfg.n_views,
                cfg_data,
                cfg.transforms['new']['train'],
                args.n_add_classes,
                args.n_known_classes,   
                rm_global_scale=cfg.rm_global_scale.new,
                drop_seed=args.drop_seed,                                
    );

    new_val_dataset = \
        getattr(
            importlib.import_module(
                '.' + args.dataset + '_multiview', package='datasets'), 
            'Dataset')(
                'val',
                args.split_type,
                cfg.n_views,
                cfg_data,
                cfg.transforms['new']['train'],
                args.n_add_classes,
                args.n_known_classes,   
                rm_global_scale=cfg.rm_global_scale.new,
                drop_seed=args.drop_seed,                                                
    );    


    old_cfg_data = getattr(importlib.import_module(
                    '.' + args.dataset, 
                    package='configs.datasets'),
                'Config_Data')(args.inverted_data_dir);   

    old_train_dataset = \
        getattr(
            importlib.import_module(
                '.' + args.dataset + '_multiview_mi', package='datasets'), 
            'Dataset')(
                'train',
                args.split_type,
                cfg.n_views,
                old_cfg_data,
                cfg.transforms['mi']['train'],
                n_add_classes=args.n_known_classes,
                n_known_classes=0,
                rm_global_scale=cfg.rm_global_scale.mi,
                is_inverted=True,
                drop_seed=args.drop_seed,                                                
    );    

    old_val_dataset = \
        getattr(
            importlib.import_module(
                '.' + args.dataset + '_multiview_mi', package='datasets'), 
            'Dataset')(
                'val',
                args.split_type,
                cfg.n_views,
                old_cfg_data,
                cfg.transforms['mi']['train'],
                n_add_classes=args.n_known_classes,
                n_known_classes=0,
                rm_global_scale=cfg.rm_global_scale.mi,
                is_inverted=True,
                drop_seed=args.drop_seed,                
    );

    # merge datasets, hard to validate
    old_train_dataset.merge_dataset(old_val_dataset);
    new_train_dataset.merge_dataset(new_val_dataset);


    old_train_sampler = None if not is_distributed else \
        torch.utils.data.distributed.DistributedSampler(
            old_train_dataset, shuffle=True, 
            drop_last=True if n_gpus>1 else False);
    new_train_sampler = None if not is_distributed else \
        torch.utils.data.distributed.DistributedSampler(
            new_train_dataset, shuffle=True, 
            drop_last=True if n_gpus>1 else False);            


    old_train_loader = torch.utils.data.DataLoader(
        old_train_dataset, 
        batch_size=cfg.batch_size.mi, 
        shuffle=(old_train_sampler is None),
        num_workers=cfg.workers, 
        pin_memory=True, 
        sampler=old_train_sampler, 
        drop_last=True if n_gpus>1 else False,
    );

    new_train_loader = torch.utils.data.DataLoader(
        new_train_dataset, 
        batch_size=cfg.batch_size.new, 
        shuffle=(new_train_sampler is None),
        num_workers=cfg.workers, 
        pin_memory=True, 
        sampler=new_train_sampler, 
        drop_last=True if n_gpus>1 else False,
    );


    optimizer, scheduler = optimizer_defs.get_optimizer_scheduler(
                        model, edict({**cfg.optimizer, 
                            'total_epochs': cfg.total_epochs, 
                            'n_steps_per_epoch': len(new_train_loader),
                        }), );

    old_criteria = loss_defs.get_losses(
                            cfg.loss.mi, 
                            cfg_data.get_n_classes(args.split_type), 
                        );

    new_criteria = loss_defs.get_losses(
                            cfg.loss.new, 
                            cfg_data.get_n_classes(args.split_type), 
                        );                        

    # print(optimizer);
    # print(scheduler);
    # print(criteria);
    # sys.exit();

    resume_checkpoint_path = utils.get_last_checkpoint_path(args.log_dir);
    if resume_checkpoint_path :
        print(f"=> loading checkpoint {resume_checkpoint_path}");
        checkpoint = torch.load(resume_checkpoint_path, map_location=torch.device('cpu'));
        args.start_epoch = checkpoint['epoch'] + 1;
        
        if args.start_epoch >= cfg.total_epochs :
            print(f"Start epoch {args.start_epoch} is greater than total epochs {cfg.total_epochs}");
            sys.exit();

        utils.load_state_dict_single(checkpoint['state_dict'], model, optimizer, scheduler, );
        print(f"=> loaded checkpoint for epoch {checkpoint['epoch']}");

        del checkpoint;
    
    # if no ckpt found for resuming, start from pretrained model
    else :
        # resume_checkpoint_path = utils.get_last_checkpoint_path(args.pretrain_dir, max_epoch=10000);
        resume_checkpoint_path = utils.get_best_model_path(args.pretrain_dir);
        if resume_checkpoint_path :
            print(f"=> loading pretrained checkpoint {resume_checkpoint_path}");
            checkpoint = torch.load(resume_checkpoint_path, map_location=torch.device('cpu'));
            args.start_epoch = 1;
            
            # # don't want the scheduler
            # utils.load_state_dict_single(checkpoint['state_dict'], model, optimizer );
            # # # reset lr
            # # for g in optimizer.param_groups :
            # #     g['lr'] = cfg.optimizer.lr;

            utils.load_state_dict_single(checkpoint['state_dict'], model );

            print(f"=> loaded pretrained checkpoint for epoch {checkpoint['epoch']}");

            del checkpoint;

        else :
            args.start_epoch = 1;
            print("=> no checkpoint found for resuming.");

    # transfer models
    model.cuda(args.gpu);

    if is_distributed :
        model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu]);

    # transfer optimizers and schedulers
    optimizer_defs.optimizer_to_cuda(optimizer, args.gpu);
    optimizer_defs.scheduler_to_cuda(scheduler, args.gpu);

    if args.gpu == 0 :
        train_logger = utils.TensorBoardLogger(osp.join(args.log_dir, 'train'));
        val_logger = utils.TensorBoardLogger(osp.join(args.log_dir, 'val'));

        # epoch, train, val bars
        print('Printing progress info for GPU 0 only ...');
        ebar = tqdm(total=cfg.total_epochs - args.start_epoch + 1, leave=True, desc='epoch', dynamic_ncols=False);
        tbar = tqdm(total=len(new_train_loader), leave=True, desc='train', dynamic_ncols=False);

    step_per_epoch = False;
    if 'scheduler' in cfg.optimizer :
        if 'step_per_epoch' in cfg.optimizer.scheduler :
            step_per_epoch = cfg.optimizer.scheduler.step_per_epoch;

    for epoch in range(args.start_epoch, cfg.total_epochs + 1) :

        torch.cuda.empty_cache();

        train(tbar if args.gpu==0 else None, 
            epoch, model, 
            new_train_loader, new_train_sampler, 
            old_train_loader, old_train_sampler, 
            optimizer, scheduler, 
            new_criteria, old_criteria,
            args, cfg, cfg_data,
            train_logger if args.gpu==0 else None,
        );


        train_logger.flush();        

        # not validating
        # assuming the last epoch is the best one for finetuning
        is_best = True; 

        # save model            
        state_dict = utils.get_state_dict_single(
                model, optimizer, scheduler, 
                is_distributed, 
        );

        utils.save_checkpoint(
            args.log_dir,
            { 
                'epoch': epoch, 
                'state_dict': state_dict, 
            },
            epoch,
            save_last_only=args.save_last_only,
            is_best=is_best,                
        );    

        if step_per_epoch :
            optimizer_defs.step_scheduler(scheduler);

        if args.gpu == 0 :
            ebar.update();
            ebar.set_postfix(dict(epoch=epoch));            

    if args.gpu == 0 :
        tbar.close();
        ebar.close();

        train_logger.close();


def train(tbar, epoch, model, 
        new_train_loader, new_train_sampler, 
        old_train_loader, old_train_sampler,
        optimizer, scheduler, 
        new_criteria, old_criteria,
        args, cfg, cfg_data,
        train_logger, ) :

    losses = edict();
    for name in old_criteria : losses[name] = utils.AverageMeter();
    for name in new_criteria : losses[name] = utils.AverageMeter();

    n_batches = len(new_train_loader);
    steps_done = (epoch-1) * n_batches;

    # set to train mode
    # model.train();
    model.pretrain();

    # set epochs
    if old_train_sampler is not None :
        old_train_sampler.set_epoch(old_train_sampler.epoch + 1);
    if new_train_sampler is not None :
        new_train_sampler.set_epoch(new_train_sampler.epoch + 1);        

    if args.gpu == 0 :
        log_freq = cfg.log_freq.train;

        tbar.reset(total=n_batches);
        tbar.refresh();

    step_per_batch = False;
    if 'scheduler' in cfg.optimizer :
        if 'step_per_batch' in cfg.optimizer.scheduler :
            step_per_batch = cfg.optimizer.scheduler.step_per_batch;

    new_iter_loader = iter(new_train_loader);
    old_iter_loader = iter(old_train_loader);
    bi = 1;

    while bi <= n_batches :
        optimizer.zero_grad();

        # load new data
        try :
            data_new = next(new_iter_loader);
        except :
            if new_train_sampler is not None :
                new_train_sampler.set_epoch(new_train_sampler.epoch + 1);
            new_iter_loader = iter(new_train_loader);
            data_new = next(new_iter_loader);
        # transfer data to gpu
        utils.tensor_dict_to_cuda(data_new, args.gpu);

        # load old data
        try :
            data_old = next(old_iter_loader);
        except :
            if old_train_sampler is not None :
                old_train_sampler.set_epoch(old_train_sampler.epoch + 1);
            old_iter_loader = iter(old_train_loader);
            data_old = next(old_iter_loader);
        # transfer data to gpu
        utils.tensor_dict_to_cuda(data_old, args.gpu);

        # new contrastive output
        pts_new, target_new = data_new.pts, data_new.label;
        bs, n_views = pts_new.shape[:2];
        pts_new = pts_new.view(bs * n_views, *pts_new.shape[2:]);
        output_new = model.forward_contrastive(pts_new);
        output_new = output_new.view(bs, n_views, *output_new.shape[1:]);
        mask_new = torch.ones((bs * n_views,), dtype=bool, device=output_new.device);

        # old contrastive output
        pts_old, target_old, target_feature_old, pred_feature_old = \
                data_old.pts, data_old.label, \
                data_old.label_feature, data_old.pred_feature;
        bs, n_views = pts_old.shape[:2];
        pts_old = pts_old.view(bs * n_views, *pts_old.shape[2:]);
        target_feature_old = target_feature_old.view(bs * n_views, *target_feature_old.shape[2:]);
        pred_feature_old = pred_feature_old.view(bs * n_views, *pred_feature_old.shape[2:]);

        output_old = model.forward_contrastive(pts_old);
        output_old = output_old.view(bs, n_views, *output_old.shape[1:]);
        mask_old = torch.zeros((bs * n_views,), dtype=bool, device=output_old.device);

        # merge old and new outputs
        output = torch.cat((output_new, output_old.detach().clone()), dim=0);
        target = torch.cat((target_new, target_old.detach().clone()), dim=0);
        mask = torch.cat((mask_new, mask_old), dim=0);

        new_old_loss_tensors = [];

        for lname in new_criteria :
            lfunc = new_criteria[lname].func;
            lweight = new_criteria[lname].weight;
            lval = lfunc(output, target, mask);
            losses[lname].update(lval.item(), output.size(0) );
            new_old_loss_tensors.append(lweight * lval);

        new_old_loss = sum(new_old_loss_tensors);

        # ============= old (mi) training step ============ #
        # mimic pred not target
        target_feature_old = pred_feature_old;

        old_loss_tensors = [];

        for lname in old_criteria :
            if lname == 'snapshot' :
                bs, n_views = output_old.shape[:2];
                output_old = output_old.view(-1, *output_old.shape[2:]);
                lfunc = old_criteria[lname].func;
                lweight = old_criteria[lname].weight;
                lval = lfunc(output_old, target_feature_old);
                losses[lname].update(lval.item(), output_old.size(0) );
                old_loss_tensors.append(lweight * lval);
                output_old = output_old.view(bs, n_views, *output_old.shape[1:]);
                continue;

            lfunc = old_criteria[lname].func;
            lweight = old_criteria[lname].weight;
            lval = lfunc(output_old, target_old);
            losses[lname].update(lval.item(), output.size(0) );
            old_loss_tensors.append(lweight * lval);                       

        old_loss = sum(old_loss_tensors);

        loss = new_old_loss + old_loss;
        # ================================================ #
        

        loss.backward();
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01);
        optimizer.step();

        if step_per_batch :
            optimizer_defs.step_scheduler(scheduler);

        # # # print(bi);
        # if bi > 10 :
        #     break;

        steps_done += 1;

        if args.gpu == 0 :

            if bi % log_freq == 0 :

                # hyperparam update
                train_logger.update(
                    {'learning_rate': optimizer.param_groups[0]['lr']},
                    step=steps_done, prefix="stepwise");

                # loss update
                train_logger.update(
                    { ltype: lmeter.avg for ltype, lmeter in losses.items() },
                    step=steps_done, prefix="loss");

                train_logger.flush();              

            tbar.update();
            tbar.set_postfix({
                    'it': bi,
                    'loss': loss.item(), 
            });
            tbar.refresh();

        bi += 1;



if __name__ == '__main__' :
    main();