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

parser.add_argument('--n_add_classes', type=int, required=True, 
                    help='#classes to add (-1) to keep all.');
parser.add_argument('--n_known_classes', type=int, required=True, 
                    help='#classes known before.');
parser.add_argument('--drop_seed', type=int, required=True, 
                    help='seed used for class dropping.');                    

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

    assert cfg.batch_size % n_gpus == 0, \
            f"Batch size {cfg.batch_size} must be divisible by #GPUs {n_gpus}";
    assert cfg.workers % n_gpus == 0, \
            f"Workers {cfg.workers} must be divisible by #GPUs {n_gpus}";

    is_distributed = n_gpus > 1;

    # split batch size and workers across gpus
    cfg.batch_size = cfg.batch_size // n_gpus;
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

    train_dataset = \
        getattr(
            importlib.import_module(
                '.' + args.dataset + '_multiview', package='datasets'), 
            'Dataset')(
                'train',
                args.split_type,
                cfg.n_views,
                cfg_data,
                cfg.transforms['train'],
                args.n_add_classes,
                args.n_known_classes,  
                rm_global_scale=cfg.rm_global_scale,
                drop_seed=args.drop_seed,                                
    );

    val_dataset = \
        getattr(
            importlib.import_module(
                '.' + args.dataset, package='datasets'), 
            'Dataset')(
                'val',
                args.split_type,
                cfg_data,
                cfg.transforms['val'],
                args.n_add_classes,
                args.n_known_classes,
                rm_global_scale=cfg.rm_global_scale,
                drop_seed=args.drop_seed,                                    
    );    

    train_sampler = None if not is_distributed else \
        torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, 
            drop_last=True if n_gpus>1 else False);
    val_sampler = None if not is_distributed else \
        torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, 
            drop_last=True if n_gpus>1 else False);

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=(train_sampler is None),
        num_workers=cfg.workers, 
        pin_memory=True, 
        sampler=train_sampler, 
        drop_last=True if n_gpus>1 else False,
    );

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=(val_sampler is None),
        num_workers=cfg.workers, 
        pin_memory=True, 
        sampler=val_sampler, 
        drop_last=True if n_gpus>1 else False,
    );

    optimizer, scheduler = optimizer_defs.get_optimizer_scheduler(
                        model, edict({**cfg.optimizer, 
                            'total_epochs': cfg.total_epochs, 
                            'n_steps_per_epoch': len(train_loader),
                        }), );

    criteria = loss_defs.get_losses(
                            cfg.loss, 
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
        tbar = tqdm(total=len(train_loader), leave=True, desc='train', dynamic_ncols=False);
        vbar = tqdm(total=len(val_loader), leave=True, desc='val', dynamic_ncols=False);

    step_per_epoch = False;
    if 'scheduler' in cfg.optimizer :
        if 'step_per_epoch' in cfg.optimizer.scheduler :
            step_per_epoch = cfg.optimizer.scheduler.step_per_epoch;

    for epoch in range(args.start_epoch, cfg.total_epochs + 1) :

        torch.cuda.empty_cache();

        train(tbar if args.gpu==0 else None, 
            epoch, model, 
            train_loader, train_sampler, 
            optimizer, scheduler, criteria, 
            args, cfg, cfg_data,
            train_logger if args.gpu==0 else None,
        );


        train_logger.flush();        

        if (epoch % args.save_epoch_freq == 0) and (args.gpu == 0) :
            measures = validate(vbar if args.gpu==0 else None, 
                epoch, model, 
                val_loader,
                criteria,
                args, cfg, cfg_data,
                val_logger if args.gpu==0 else None,
            ); 

            is_best = best_measure_info.func(
                        measures[best_measure_info.tag],
                        best_measure_info.val, );
            if is_best :
                best_measure_info.val = measures[best_measure_info.tag];            

            val_logger.flush();    

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
                    'best_measure_tag': best_measure_info.tag,
                    'best_measure': best_measure_info.val,                     
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
        val_logger.close();


def train(tbar, epoch, model, 
        train_loader, train_sampler, 
        optimizer, scheduler, criteria, 
        args, cfg, cfg_data,
        train_logger, ) :

    losses = edict({
        name: utils.AverageMeter() for name in criteria
    });
   
    n_batches = len(train_loader);
    steps_done = (epoch-1) * n_batches;

    # set to train mode
    # model.train();
    model.pretrain();

    # set epochs
    if train_sampler is not None :
        train_sampler.set_epoch(train_sampler.epoch + 1);

    if args.gpu == 0 :
        log_freq = cfg.log_freq.train;

        tbar.reset(total=n_batches);
        tbar.refresh();

    step_per_batch = False;
    if 'scheduler' in cfg.optimizer :
        if 'step_per_batch' in cfg.optimizer.scheduler :
            step_per_batch = cfg.optimizer.scheduler.step_per_batch;

    iter_loader = iter(train_loader);
    bi = 1;
    while bi <= n_batches :
        data = next(iter_loader);
        # transfer data to gpu
        utils.tensor_dict_to_cuda(data, args.gpu);

        pts, target = data.pts, data.label;
        bs, n_views = pts.shape[:2];
        pts = pts.view(bs * n_views, *pts.shape[2:]);
        output = model.forward_contrastive(pts);
        output = output.view(bs, n_views, *output.shape[1:]);

        loss_tensors = [];

        for lname in criteria :
            lfunc = criteria[lname].func;
            lweight = criteria[lname].weight;
            lval = lfunc(output, target);
            losses[lname].update(lval.item(), output.size(0) );
            loss_tensors.append(lweight * lval);

        loss = sum(loss_tensors);

        optimizer.zero_grad();
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

@torch.no_grad()
def load_proto_mat(proto_mean_d, n_classes, gpu) :
    # load prototypes
    dtype = torch.get_default_dtype();

    for label in proto_mean_d :
        dim = proto_mean_d[label].shape[-1];
        break;

    proto_mean = np.zeros((n_classes, dim), dtype=np.float32);
    for k in proto_mean_d :
        proto_mean[k] = np.copy(proto_mean_d[k]);

    proto_mean = torch.from_numpy(proto_mean).to(dtype).to(gpu);
    return proto_mean;

@torch.no_grad()
def get_class_ids(features, proto_mean, drop_list) :
    assert not proto_mean.requires_grad;

    # features (n x d); 
    # proto_mean (c x d); 

    n, d = features.shape;
    c, _ = proto_mean.shape;
    max_val = torch.finfo(torch.get_default_dtype()).max;

    dist_ = torch.cdist(features, proto_mean.to(features.device));

    # dist_others = torch.sum(dist_, dim=1);
    # dist_others = dist_others - dist_.data[:, drop_list];
    # dist_.data[:, drop_list] = max_val;
    # dist_min, pred_ids = torch.min(dist_, dim=1);
    # dist_others = dist_others - dist_min.data;

    if drop_list is not None :
        dist_.data[:, drop_list] = max_val;
    dist_knn, _ = torch.topk(dist_, k=2, dim=1, largest=False);
    dist_dif = dist_knn[:, 1] - dist_knn[:, 0];
    _, pred_ids = torch.min(dist_, dim=1);

    return dist_dif, pred_ids;


@torch.no_grad()
def validate(vbar,  
        epoch, model, 
        val_loader,
        criteria,
        args, cfg, cfg_data,
        val_logger, ) :
   
    n_classes = cfg_data.get_n_classes(args.split_type);
    drop_list = utils.get_drop_class_list(n_classes, args.n_add_classes, args.n_known_classes);

    dist_meter = utils.AverageMeter();
    iou_meter = utils.PIoUMeter(
                    n_classes, 
                    cfg_data.label_to_name[args.split_type] );  

    # set to eval mode
    model.eval();

    n_batches = len(val_loader);

    n_gpus = torch.cuda.device_count();
    is_gather = n_gpus>1;
    if is_gather :
        output_l = None;
        target_l = None;
        dist_l = None;

    if args.gpu == 0 :
        vbar.reset(total=len(val_loader));
        vbar.refresh();

    iter_loader = iter(val_loader);
    bi = 1;

    proto_mean, samples_per_class = {}, {};
    while bi <= n_batches :
        data = next(iter_loader);
        # transfer data to gpu
        utils.tensor_dict_to_cuda(data, args.gpu);

        pts, target = data.pts, data.label;

        output = model.forward_feature(pts);
        output = output.data.cpu().numpy();
        for k in range(output.shape[0]) :
            feat = output[k];
            label = data.label[k].item();

            dim, dtype = feat.size, feat.dtype;
            # init class-wise mean
            if label not in proto_mean :
                proto_mean[label] = np.zeros((1, dim), dtype=dtype);
                samples_per_class[label] = 0;
            
            proto_mean[label] += feat[None, :];
            samples_per_class[label] += 1;

        vbar.update();            
        vbar.refresh();

        bi += 1;

    # normalize 
    for label in samples_per_class :
        proto_mean[label] /= samples_per_class[label];

    proto_mean = load_proto_mat(proto_mean, n_classes, args.gpu);

    if args.gpu == 0 :
        vbar.reset(total=len(val_loader));
        vbar.refresh();

    iter_loader = iter(val_loader);
    bi = 1;

    while bi <= n_batches :
        data = next(iter_loader);
        # transfer data to gpu
        utils.tensor_dict_to_cuda(data, args.gpu);

        pts, target = data.pts, data.label;
        features = model.forward_feature(pts);
        dist_dif, output = get_class_ids(features, proto_mean, drop_list);

        # distributed gather for multi gpu
        if is_gather : 
            if output_l is None :
                output_l = [torch.zeros_like(output) for _ in range(n_gpus)];
                target_l = [torch.zeros_like(target) for _ in range(n_gpus)];
                dist_l = [torch.zeros_like(dist_dif) for _ in range(n_gpus)];

            dist.all_gather(output_l, output);
            dist.all_gather(target_l, target);
            dist.all_gather(dist_l, dist_dif);

            output = torch.cat(output_l, dim=0);
            target = torch.cat(target_l, dim=0);
            dist_dif = torch.cat(dist_l, dim=0);
        # ================================= #

        dist_mean = dist_dif.mean().item();
        dist_meter.update(dist_mean, dist_dif.size(0));
        iou_meter.update(output, target, take_max=False);

        if args.gpu == 0 :
            vbar.update();
            vbar.set_postfix({
                'it': bi,
                'distance': dist_mean, 
            });                
            vbar.refresh();

        bi += 1;

        # # # print(bi);
        # if bi > 10 :
        #     break;        

    if args.gpu == 0 :
        iou_all = iou_meter.stats();

        # loss update
        val_logger.update(
            { 'proto_dist': dist_meter.avg },
            step=epoch, prefix="loss");

        # measures update
        val_logger.update({
            'mean': iou_all.iou,
            }, step=epoch, prefix="iou" );

        iou_mean = iou_all.pop('iou');
        val_logger.update(
            { cname: cval for cname, cval in iou_all.items() },
            step=epoch, prefix="iou_per_class" );                     

        val_logger.flush(); 

        return {
            'iou': iou_mean,
            'dist_dif': dist_meter.avg,
        };


if __name__ == '__main__' :
    main();