import sys
import os, os.path as osp

import argparse

from tqdm import tqdm
from typing import Set

if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))
sys.path.append(osp.join('..', '..'))

from configs.datasets.hgr_shrec_2017 import Config_Data
from utils.stdio import *

parser = argparse.ArgumentParser(description='Generate train/val/test splits for HGR SHREC 2017.')
parser.add_argument('--root_dir', default='/ogr_cmu/data/SHREC_2017', type=str, 
                    help='root directory containing the dataset.')

args = parser.parse_args()
print_argparser_args(args)


def list_subjects_single_subset(
    root_dir: str, 
    split_filepath: str,
) -> Set[int] :

    n_samples = get_nlines_in_file(split_filepath)
    fhand = get_file_handle(split_filepath, 'r')

    subject_l = set()
    for line in tqdm( fhand, total=n_samples ) :
      
        line = line.strip()
        # sub_path, label, size_seq, id_subject = line
        _, _, _, id_subject = line.split(',')
        id_subject = int(id_subject)
        subject_l.add(id_subject)

    fhand.close()
    
    return subject_l



def main() :
    global args
    cfg = Config_Data(args.root_dir)    

    split_type = 'agnostic'
    subject_l = {}
    split_file_l = {}
    for mode in cfg.split_files :
        
        print(f"'{split_type}' split, '{mode}' set ...")
        split_file = cfg.get_split_filepath(split_type, mode)
        split_file_l[mode] = osp.basename(split_file)
        subject_l[mode] = list_subjects_single_subset(
                args.root_dir, 
                split_file,
        )

        print(f'subject list ({len(subject_l[mode])}) =', sorted(list(subject_l[mode])))
        print()


    for m1 in cfg.split_files :
        for m2 in cfg.split_files :
            if m1 == m2 : 
                continue

            if split_file_l[m1] == split_file_l[m2] :
                continue

            assert (subject_l[m1].isdisjoint(subject_l[m2])), \
                f"{m1} --> {subject_l[m1]}, {m2} --> {subject_l[m2]}"


if __name__ == "__main__" :
    main()
    sys.exit()