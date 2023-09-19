import sys
import os, os.path as osp
import argparse


if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))

SUB_DIR_LEVEL = 1 # level of this subdirectory w.r.t. root of the code
sys.path.append(osp.join(*(['..'] * SUB_DIR_LEVEL)))

import utils

parser = argparse.ArgumentParser(description='Summarize results.')
parser.add_argument('--root_log_dir', type=str, default=1, required=False, help='')
parser.add_argument('--n_trials', type=int, default=3, required=False, help='')
parser.add_argument('--n_tasks', type=int, default=7, required=False, help='')


def main() :
    args = parser.parse_args()

    print('--------------------Summarizing results--------------------')
    utils.summarize_results(args.root_log_dir, args.n_trials, args.n_tasks)


if __name__ == '__main__' :
    main()