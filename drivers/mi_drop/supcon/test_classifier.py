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

parser.add_argument('--proto_fname', type=str, default='proto.pkl', 
                    help='name of the proto file.');
parser.add_argument('--clf_fname', type=str, default='classifier.pkl', 
                    help='name of the classifier file.');

parser.add_argument('--n_add_classes', type=int, required=True, 
                    help='#classes to add (-1) to keep all.');
parser.add_argument('--n_known_classes', type=int, required=True, 
                    help='#classes known before.');                    
parser.add_argument('--drop_seed', type=int, required=True, 
                    help='seed used for class dropping.');

parser.add_argument('--use_reduced', action='store_true', 
                    help='whether to use the reduced representation.');

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


    mode = 'test';
    # define dataset
    test_dataset = \
        getattr(
            importlib.import_module(
                '.' + args.dataset, package='datasets'), 
            'Dataset')(
                mode,
                args.split_type,
                cfg_data,
                cfg.transforms[mode],
                args.n_add_classes,
                args.n_known_classes,                 
                rm_global_scale=cfg.rm_global_scale,
                drop_seed=args.drop_seed,                    
    );        

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False,
        num_workers=cfg.workers, 
        pin_memory=True, 
        sampler=None, 
        drop_last=False,            
    );

    # load checkpoint
    best_model_path = utils.get_best_model_path(args.log_dir);
    
    assert best_model_path is not None, \
        f"Best model checkpoint not found in the log directory {args.log_dir}";
    print(f"=> loading checkpoint {best_model_path}");
    checkpoint = torch.load(best_model_path, map_location=torch.device('cpu'));

    epoch = checkpoint['epoch'];
    utils.load_state_dict_single(checkpoint['state_dict'], model );
    print(f"=> loaded checkpoint for epoch {checkpoint['epoch']}");

    del checkpoint;

    # transfer models
    model.cuda(args.gpu);


    test_validate(
        model, 
        test_loader,
        args, cfg, cfg_data,
    ); 



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
def get_class_ids(features, proto_mean, drop_list=None) :
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
def load_proto(fpath, n_classes, is_reduced) :
    # load prototypes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    dtype = torch.get_default_dtype();

    data = utils.load_pickle(fpath);
    data = data['reduced' if is_reduced else 'raw'];
    proto_mean = data['mean'];
    proto_var = data['var'];

    if is_reduced :
        proto_eig_vecs = data['eig_vecs'];
        return proto_mean, proto_var, proto_eig_vecs;    
    else :
        # proto_var = utils.add_dummy_variances(proto_var);
        for k in proto_mean :
            proto_mean[k] = proto_mean[k];
        return proto_mean, proto_var;


# @torch.no_grad()
# def test_validate( 
#     model, 
#     test_loader,
#     args, cfg, cfg_data
# ) :
   
#     is_reduced = args.use_reduced;   
#     n_classes = cfg_data.get_n_classes(args.split_type);
#     protos = load_proto(
#             osp.join(args.log_dir, args.proto_fname), 
#             n_classes, is_reduced=is_reduced,
#     );    

#     if is_reduced :
#         proto_mean, _, _ = protos;
#     else :
#         proto_mean, _ = protos;    

#     iou_meter = utils.PIoUMeter(
#                     n_classes, 
#                     cfg_data.label_to_name[args.split_type] );  

#     # set to eval mode
#     model.eval();

#     n_batches = len(test_loader);

#     n_gpus = torch.cuda.device_count();
#     is_gather = n_gpus>1;
#     if is_gather :
#         output_l = None;
#         target_l = None;

#     if args.gpu == 0 :
#         vbar = tqdm(total=len(test_loader), leave=True, desc='val', dynamic_ncols=False);
#         vbar.refresh();

#     iter_loader = iter(test_loader);
#     bi = 1;

#     proto_mean = load_proto_mat(proto_mean, n_classes, args.gpu);

#     if args.gpu == 0 :
#         vbar.reset(total=len(test_loader));
#         vbar.refresh();

#     iter_loader = iter(test_loader);
#     bi = 1;

#     while bi <= n_batches :
#         data = next(iter_loader);
#         # transfer data to gpu
#         utils.tensor_dict_to_cuda(data, args.gpu);

#         pts, target = data.pts, data.label;
#         features = model.forward_feature(pts);
#         dist_dif, output = get_class_ids(features, proto_mean);

#         # distributed gather for multi gpu
#         if is_gather : 
#             if output_l is None :
#                 output_l = [torch.zeros_like(output) for _ in range(n_gpus)];
#                 target_l = [torch.zeros_like(target) for _ in range(n_gpus)];

#             dist.all_gather(output_l, output);
#             dist.all_gather(target_l, target);

#             output = torch.cat(output_l, dim=0);
#             target = torch.cat(target_l, dim=0);
#         # ================================= #

#         iou_meter.update(output, target, take_max=False);

#         if args.gpu == 0 :
#             vbar.update();
#             vbar.set_postfix({
#                 'it': bi,
#             });                
#             vbar.refresh();

#         bi += 1;

#         # # # print(bi);
#         # if bi > 10 :
#         #     break;        

#     if args.gpu == 0 :
#         iou_all = iou_meter.stats();
#         utils.print_ious(
#             iou_all, 
#             cfg_data.get_n_classes(args.split_type), 
#             cfg_data.label_to_name[args.split_type],
#         );


@torch.no_grad()
def test_validate( 
    model, 
    test_loader,
    args, cfg, cfg_data
) :
   
    n_classes = cfg_data.get_n_classes(args.split_type);

    iou_meter = utils.PIoUMeter(
                    n_classes, 
                    cfg_data.label_to_name[args.split_type] );  

    clf = utils.load_pickle(osp.join(args.log_dir, args.clf_fname));

    # set to eval mode
    model.eval();

    n_batches = len(test_loader);

    n_gpus = torch.cuda.device_count();
    is_gather = n_gpus>1;
    if is_gather :
        output_l = None;
        target_l = None;

    if args.gpu == 0 :
        vbar = tqdm(total=len(test_loader), leave=True, desc='val', dynamic_ncols=False);
        vbar.refresh();

    iter_loader = iter(test_loader);
    bi = 1;

    if args.gpu == 0 :
        vbar.reset(total=len(test_loader));
        vbar.refresh();

    iter_loader = iter(test_loader);
    bi = 1;

    while bi <= n_batches :
        data = next(iter_loader);
        # transfer data to gpu
        utils.tensor_dict_to_cuda(data, args.gpu);

        pts, target = data.pts, data.label;
        features = model.forward_feature(pts);
        features = features.data.cpu().numpy();
        output = utils.predict_classifier(clf, features);
        output = torch.from_numpy(output).long().to(target.device);

        # distributed gather for multi gpu
        if is_gather : 
            if output_l is None :
                output_l = [torch.zeros_like(output) for _ in range(n_gpus)];
                target_l = [torch.zeros_like(target) for _ in range(n_gpus)];

            dist.all_gather(output_l, output);
            dist.all_gather(target_l, target);

            output = torch.cat(output_l, dim=0);
            target = torch.cat(target_l, dim=0);
        # ================================= #

        iou_meter.update(output, target, take_max=False);

        if args.gpu == 0 :
            vbar.update();
            vbar.set_postfix({
                'it': bi,
            });                
            vbar.refresh();

        bi += 1;

        # # # print(bi);
        # if bi > 10 :
        #     break;        

    vbar.close();

    if args.gpu == 0 :
        print(f"Accuracy = {iou_meter.accuracies():.1f} %");



if __name__ == '__main__' :
    main();