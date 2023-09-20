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

SUB_DIR_LEVEL = 3; # level of this subdirectory w.r.t. root of the code
sys.path.append(osp.join(*(['..'] * SUB_DIR_LEVEL)));

import model_defs
import utils

parser = argparse.ArgumentParser(description='Saving the classifier.')
parser.add_argument('--dataset', type=str, required=True, help='name of the dataset.');
parser.add_argument('--subset', type=str, required=True, help='name of the data subset (train/val).');
parser.add_argument('--split_type', type=str, required=True, help='type of data split (if applicable).');
parser.add_argument('--cfg_file', type=str, required=True, help='config file to load experimental parameters.');
parser.add_argument('--root_dir', type=str, required=True, help='root directory containing the dataset.');
parser.add_argument('--log_dir', type=str, required=True, help='directory for logging.');
parser.add_argument('--clf_fname', type=str, default='classifier.pkl', 
                    help='name of the classifier file.');       

parser.add_argument('--n_add_classes', type=int, required=True, 
                    help='#classes to add (-1) to keep all.');
parser.add_argument('--n_known_classes', type=int, required=True, 
                    help='#classes known before.');
parser.add_argument('--drop_seed', type=int, required=True, 
                    help='seed used for class dropping.');

parser.add_argument('--eig_var_exp', type=float, required=True, 
                    help='variance explanation factor for eigen decomposition.'); 


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

    mode = 'test' + args.subset;
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

    if args.subset == 'val' :
        extra_test_dataset = \
            getattr(
                importlib.import_module(
                    '.' + args.dataset, package='datasets'), 
                'Dataset')(
                    'testtrain',
                    args.split_type,
                    cfg_data,
                    cfg.transforms[mode],
                    args.n_add_classes,
                    args.n_known_classes,                   
                    rm_global_scale=cfg.rm_global_scale,
                    drop_seed=args.drop_seed,                    
        );            
                
        test_dataset.merge_dataset(extra_test_dataset);


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

    test(
        epoch, model, 
        test_loader,
        args, cfg, cfg_data,
    ); 


@torch.no_grad()
def test(  
    epoch, model, 
    test_loader,
    args, cfg, cfg_data, 
) :
   
    from pprint import pprint

    label_to_name = cfg_data.label_to_name[args.split_type];

    # set to eval mode
    model.eval();

    proto_mean, proto_var, samples_per_class = {}, {}, {};
    dim, dtype = None, None;
    n_batches = len(test_loader);

    # ============= compute class-wise mean ================ #
    vbar = tqdm(total=len(test_loader), leave=True, desc='mean', dynamic_ncols=False);
    vbar.refresh();
    iter_loader = iter(test_loader);
    bi = 1;

    X_train, y_train = [], [];

    while bi <= n_batches :
        data = next(iter_loader);
        # transfer data to gpu
        utils.tensor_dict_to_cuda(data, args.gpu);

        pts, target = data.pts, data.label;

        output = model.forward_feature(pts);
        output = output.data.cpu().numpy();

        # for k in range(output.shape[0]) :
        #     feat = output[k];
        #     label = data.label[k].item();

        #     dim, dtype = feat.size, feat.dtype;
        #     # init class-wise mean
        #     if label not in proto_mean :
        #         proto_mean[label] = np.zeros((1, dim), dtype=dtype);
        #         samples_per_class[label] = 0;
            
        #     proto_mean[label] += feat[None, :];
        #     samples_per_class[label] += 1;

        label = data.label.data.cpu().numpy();
        X_train.append(output);
        y_train.append(label);

        vbar.update();            
        vbar.refresh();

        bi += 1;

    vbar.close();

    X_train = np.concatenate(X_train, axis=0);
    y_train = np.concatenate(y_train, axis=0);

    clf = utils.get_svm_classifier('rbf');
    print("Fitting classifier to data ...")
    print(f"Shape of data = {X_train.shape}"); 
    utils.fit_classifier(clf, X_train, y_train);

    clf_fpath = osp.join(args.log_dir, args.clf_fname);
    print(f"Saving classifier in = {clf_fpath}");
    utils.save_pickle(clf_fpath, clf);    


if __name__ == '__main__' :
    main();