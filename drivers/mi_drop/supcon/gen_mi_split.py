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

parser = argparse.ArgumentParser(description='Merge inverted sets.')
parser.add_argument('--in_dir_l', nargs='+', required=True,
                    help='list of input directories to merge files from.');
parser.add_argument('--out_dir', type=str, required=True, help='output directory.');
parser.add_argument('--split_file_subdir', type=str, required=True, 
                    help='subdirectory containing split files.');

def process_single_dir(in_dir, out_dir, fin, fout, count_per_class) :
    # print(split_filepath);
    for line in fin :
        line = line.strip();
        fpath, class_id, seq_len, id_subj = line.split(',');

        f_subdir = osp.dirname(fpath);
        tmp_out_dir = osp.join(out_dir, f_subdir);
        os.makedirs(tmp_out_dir, exist_ok=True);
        in_fname = osp.basename(fpath);
        file_ext = osp.splitext(in_fname)[1];
        count_per_class[class_id] = count_per_class.get(class_id, -1) + 1;
        count = count_per_class[class_id];
        out_fname = str(count).zfill(6) + file_ext;
        
        in_fpath = osp.join(in_dir, fpath);
        out_fpath = osp.join(tmp_out_dir, out_fname);
        # print(f"{in_fpath} --> {out_fpath}");
        shutil.copyfile(in_fpath, out_fpath);

        fout.write(
            osp.join(f_subdir, out_fname) + ',' + \
            class_id + ',' + \
            seq_len + ',' + \
            id_subj + '\n'
        );

def gen_mi_split(in_dir_l, out_dir, split_file_subdir) :
    assert len(in_dir_l) > 0;
    print(f"Merging files from {len(in_dir_l)} directories =>");
    for i, in_dir in enumerate(in_dir_l, 1) :
        assert osp.isdir(in_dir), f"Input directory not found = {in_dir}";
        print(f"({i}) {in_dir}");
    
    print(f"to = {out_dir}\n\n");

    out_split_file_dir = osp.join(out_dir, split_file_subdir);
    utils.mkdir_rm_if_exists(out_split_file_dir);
    out_split_files = set();

    count_per_class = {};
    for in_dir in in_dir_l :
        split_file_dir = osp.join(in_dir, split_file_subdir);
        assert osp.isdir(split_file_dir), \
                f"Split file directory not found = {split_file_dir}";

        for split_file in sorted(os.listdir(split_file_dir)) :
            split_filepath = osp.join(split_file_dir, split_file);
            if not osp.isfile(split_filepath) :
                continue;

            out_split_filepath = osp.join(out_split_file_dir, split_file);
            if out_split_filepath in out_split_files :
                fout = utils.get_file_handle(out_split_filepath, 'a');
            else :
                out_split_files.add(out_split_filepath);
                fout = utils.get_file_handle(out_split_filepath, 'w+');

            fin = utils.get_file_handle(split_filepath, 'r');
            print(f"Processing split file = {split_filepath}");
            process_single_dir(in_dir, out_dir, fin, fout, count_per_class);

            fin.close();
            fout.close();


def main() :
    args = parser.parse_args();
    # utils.print_argparser_args(args);
    gen_mi_split(
        args.in_dir_l, 
        args.out_dir,
        args.split_file_subdir,
    );


if __name__ == '__main__' :
    main();