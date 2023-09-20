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

import concurrent.futures as futures

SUB_DIR_LEVEL = 3; # level of this subdirectory w.r.t. root of the code
sys.path.append(osp.join(*(['..'] * SUB_DIR_LEVEL)));

import model_defs
import utils

parser = argparse.ArgumentParser(description='Running model inversion.')
parser.add_argument('--dataset', type=str, required=True, help='name of the dataset.');
parser.add_argument('--split_type', type=str, required=True, help='type of data split (if applicable).');
parser.add_argument('--cfg_file', type=str, required=True, help='config file to load experimental parameters.');
parser.add_argument('--root_dir', type=str, required=True, help='root directory containing the dataset.');
parser.add_argument('--log_dir', type=str, required=True, help='directory for logging.');
parser.add_argument('--proto_fname', type=str, default='proto.pkl', 
                    help='name of the proto file.');
parser.add_argument('--inv_sample_subdir', type=str, required=True, 
                    help='subdirectory to save the inverted samples.');
parser.add_argument('--inv_type', type=str, required=True, 
                    help='type of inversion (proto, svm, hybrid).');
parser.add_argument('--clf_fname', type=str, default='classifier.pkl', 
                    help='name of the proto file.');  

parser.add_argument('--n_add_classes', type=int, required=True, 
                    help='#classes to add (-1) to keep all.');
parser.add_argument('--n_known_classes', type=int, required=True, 
                    help='#classes known before.');
parser.add_argument('--drop_seed', type=int, required=True, 
                    help='seed used for class dropping.');

parser.add_argument('--max_samples_per_class', type=int, required=True, 
                    help='max #samples per class to invert.');                    

parser.add_argument('--use_reduced', action='store_true', 
                    help='whether to use the reduced representation.');


def main() :
    global best_measure_info;

    args = parser.parse_args();
    utils.print_argparser_args(args);

    utils.set_seed();
   
    n_gpus = torch.cuda.device_count();
    assert n_gpus>0, "A GPU is required for execution.";

    args.gpu = 0;

    with open(args.cfg_file, 'rb') as f :
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader));
    # pprint(cfg); sys.exit();


    cfg_data = getattr(importlib.import_module(
                    '.' + args.dataset, 
                    package='configs.datasets'),
                'Config_Data')(args.root_dir);    

    torch.cuda.set_device(args.gpu);

    model = model_defs.get_model(edict({
            'n_classes': cfg_data.get_n_classes(args.split_type),
            **cfg.model
    }) );
    model_defs.print_n_params(model);
    # print(model);
    model.cuda(args.gpu); # transfer models

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

    test(
        epoch, model, 
        args, cfg, cfg_data,
    ); 


def load_proto(fpath, n_classes, n_add_classes, n_known_classes, drop_seed, is_reduced) :
    add_class_l = utils.get_add_class_list(n_add_classes, n_known_classes, n_classes, drop_seed);
    add_class_l = set(add_class_l);

    # load prototypes
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu');
    dtype = torch.get_default_dtype();

    data = utils.load_pickle(fpath);
    data = data['reduced' if is_reduced else 'raw'];
    proto_mean = data['mean'];
    proto_var = data['var'];

    for k in proto_mean :
        assert k in add_class_l;
    for k in proto_var :
        assert k in add_class_l;

    if is_reduced :
        proto_eig_vecs = data['eig_vecs'];
        return proto_mean, proto_var, proto_eig_vecs;    
    else :
        # proto_var = utils.add_dummy_variances(proto_var);
        return proto_mean, proto_var;


# @torch.no_grad()
@utils.timer
def test(  
    epoch, model, 
    args, cfg, cfg_data, 
) :
   
    is_reduced = args.use_reduced;
    # inv_sample_subdir = args.inv_sample_subdir + '_' + args.inv_type;
    # out_dir = osp.join(args.log_dir, inv_sample_subdir);
    out_dir = osp.join(args.log_dir, args.inv_sample_subdir);
    utils.mkdir_rm_if_exists(out_dir);

    from pprint import pprint

    label_to_name = cfg_data.label_to_name[args.split_type];
    n_classes = cfg_data.get_n_classes(args.split_type);

    clf = utils.load_pickle(osp.join(args.log_dir, args.clf_fname));

    # set to eval mode
    model.eval();

    protos = load_proto(
            osp.join(args.log_dir, args.proto_fname), 
            n_classes, 
            args.n_add_classes, 
            args.n_known_classes,
            args.drop_seed,
            is_reduced=is_reduced,
    );

    if is_reduced :
        proto_mean, proto_var, proto_eig_vecs = protos;
    else :
        proto_mean, proto_var = protos;

    if args.inv_type == 'svm' \
        or args.inv_type == 'proto' \
        or args.inv_type == 'proto-svm' :

        mi_func = utils.save_inverted_samples_proto_svm_parallel_v2;

    elif args.inv_type == 'random-svm' :
        mi_func = utils.save_inverted_samples_random_svm;
    
    else :
        raise NotImplementedError;

    mi_params = cfg.mi_params;
    mi_params.inv_type = args.inv_type;
    mi_params.max_samples_per_class = args.max_samples_per_class;

    mi_func(
        out_dir, 
        model, clf, 
        proto_mean, proto_var, 
        mi_params,
    );


    # create the train/val splits
    split_dir = osp.join(out_dir, 'split_files');
    utils.mkdir_rm_if_exists(split_dir);
    train_fpath = osp.join(split_dir, 'train.txt');
    val_fpath = osp.join(split_dir, 'val.txt');

    f_train = utils.get_file_handle(train_fpath, 'w+');
    f_val = utils.get_file_handle(val_fpath, 'w+');
    crop_len_dir = len(out_dir)+1;

    seq_len = model.get_input_shape()[0];
    for cid in proto_mean :
        sample_dir = osp.join(out_dir, 'class_' + str(cid));
        utils.write_train_val_splits(
            sample_dir, crop_len_dir, 
            f_train, f_val,
            cid, seq_len,
        );

    f_train.close();
    f_val.close();


if __name__ == '__main__' :
    main();