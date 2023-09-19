import sys
import os, os.path as osp
import argparse
import yaml
import importlib
from easydict import EasyDict as edict

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
torch.use_deterministic_algorithms(True)


if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))

SUB_DIR_LEVEL = 1 # level of this subdirectory w.r.t. root of the code
sys.path.append(osp.join(*(['..'] * SUB_DIR_LEVEL)))

import utils

parser = argparse.ArgumentParser(description='Gesture Recognition.')
parser.add_argument('--train', type=int, default=1, required=False, help='train (1) or testval (0) or test (-1).')
parser.add_argument('--dataset', type=str, default='hgr_shrec_2017', required=False, help='name of the dataset.')
parser.add_argument('--split_type', type=str, default='agnostic', required=False, help='type of data split (if applicable).')
parser.add_argument('--cfg_file', type=str, default='/ogr_cmu/src/configs/params/hgr_shrec_2017/Oracle-BN.yaml', required=False, help='config file to load experimental parameters.')
parser.add_argument('--root_dir', type=str, default='/ogr_cmu/data/SHREC_2017', required=False, help='root directory containing the dataset.')
parser.add_argument('--log_dir', type=str, default='/ogr_cmu/output/hgr_shrec_2017/Oracle-BN', required=False, help='directory for logging.')
parser.add_argument('--save_last_only', action='store_true', help='whether to save the last epoch only.')
parser.add_argument('--save_epoch_freq', type=int, default=1, help='epoch frequency to save checkpoints.')
parser.add_argument('--save_conf_mat', action='store_true', default=1, help='whether to save the confusion matrix.')
parser.add_argument('--gpu', type=int, default=0, help='GPU ID.')
parser.add_argument('--trial_id', type=int, default=0, help='trial_ID')


def main() :
    args = parser.parse_args()
    args.dist_url = 'tcp://127.0.0.1:' + utils.get_free_port()
    utils.print_argparser_args(args)
    utils.set_seed()
    n_gpus = torch.cuda.device_count()
    assert n_gpus>0, "A GPU is required for execution."
    main_worker(args.gpu, 1, args)

def main_worker(gpu, n_gpus, args) :
    with open(args.cfg_file, 'rb') as f :
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
    
    cfg_data = getattr(importlib.import_module('.' + args.dataset, package='configs.datasets'),
                'Config_Data')(args.root_dir)   

    is_distributed = False
    is_train = (args.train==1)

    # Learners dict
    learners_dict = {
        'base' : 'Base',
        'lwf' : 'LwF',
        'lwf_MC' : 'LwF_MC',
        'deep_inversion' : 'DeepInversion',
        'deep_inversion_gen': 'DeepInversion_gen',
        'abd': 'AlwaysBeDreaming',
        'rdfcil':'Rdfcil'
    }
    
    # Execute trial
    root_log_dir = args.log_dir
    trial_id = args.trial_id
    with open(args.cfg_file, 'rb') as f :
            cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
            
    print(f'--------------------Executing trial {trial_id+1}--------------------')
    # Create output directory
    trial_log_dir = osp.join(root_log_dir, f'trial_{trial_id+1}')
    if not osp.exists(trial_log_dir) :
        os.makedirs(trial_log_dir)
    args.log_dir = trial_log_dir

    # Create learner
    learner = getattr(importlib.import_module('.' + cfg.increm.learner.type, package='learners'),
                learners_dict[cfg.increm.learner.type])(cfg, cfg_data, args, is_train, is_distributed, n_gpus)

    if is_train :
        # Train 
        learner.train(n_trial=trial_id)

    else:
        # Evaluate
        learner.evaluate(n_trial=trial_id)
    


if __name__ == '__main__' :
    main()