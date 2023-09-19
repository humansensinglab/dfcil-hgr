import sys
import os, os.path as osp
import shutil

import argparse

from pprint import pprint
from tqdm import tqdm

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


def list_samples_per_subject_per_class(
    root_dir: str, 
    split_filepath: str,
) -> dict :

    def get_val_subject_l(subject_l, nval=20) :
        # sort by count
        nsamples_sub_l = []
        for subj, nsamples in subject_l.items() :
            nsamples_sub_l.append((nsamples, subj))
        
        nsamples_sub_l.sort()
        count = 0
        val_subject_l = []
        for nsamples, subj in nsamples_sub_l :
            count += nsamples
            val_subject_l.append(subj)
            
            if count >= nval :
                break
        
        return val_subject_l


    n_samples = get_nlines_in_file(split_filepath)
    fhand = get_file_handle(split_filepath, 'r')

    class_subject_l = {}
    for line in tqdm( fhand, total=n_samples ) :
      
        line = line.strip()
        # sub_path, label, size_seq, id_subject = line
        _, label, _, id_subject = line.split(',')
        label = int(label)
        id_subject = int(id_subject)

        if label not in class_subject_l :
            class_subject_l[label] = {}
        
        subject_l = class_subject_l[label]
        subject_l[id_subject] = subject_l.get(id_subject, 0) + 1

    fhand.close()

    val_subject_l = {}
    for cid in class_subject_l :
        val_subject_l[cid] = get_val_subject_l(class_subject_l[cid])
    
    return val_subject_l


def split_train_to_train_val(
    val_subject_l: dict,
    root_dir: str, 
    train_split_filepath: str,
    val_split_filepath: str,
) -> dict :

    def __get_tmp_filepath(file_path) :
        f1, f2 = osp.splitext(file_path)
        return f1 + '_tmp' + f2

    tmp_train_split_filepath =  __get_tmp_filepath(train_split_filepath)

    n_samples = get_nlines_in_file(train_split_filepath)
    fin = get_file_handle(train_split_filepath, 'r')
    ftrain = get_file_handle(tmp_train_split_filepath, 'w+')
    fval = get_file_handle(val_split_filepath, 'w+')

    count_train, count_val = 0, 0
    for line in tqdm( fin, total=n_samples ) :
      
        line = line.strip()
        # sub_path, label, size_seq, id_subject = line
        _, label, _, id_subject = line.split(',')
        label = int(label)
        id_subject = int(id_subject)

        if id_subject in val_subject_l[label] :
            count_val += 1
            fout = fval
        else :
            count_train += 1
            fout = ftrain

        fout.write(line + '\n')

    fin.close()
    ftrain.close()
    fval.close()

    os.remove(train_split_filepath)
    shutil.move(tmp_train_split_filepath, train_split_filepath)
    
    print(f"# (Train, Val) samples = ({count_train}, {count_val})")



def main() :
    global args
    cfg = Config_Data(args.root_dir)    

    split_type = 'agnostic'
    mode = 'train'
    print(f"'{split_type}' split, '{mode}' set ...")
    val_subject_l = list_samples_per_subject_per_class(
            args.root_dir, 
            cfg.get_split_filepath(split_type, mode),
    )

    print("Val subject list =>")
    pprint(val_subject_l)
    print()

    split_train_to_train_val(
            val_subject_l,
            args.root_dir, 
            cfg.get_split_filepath(split_type, 'train'),
            cfg.get_split_filepath(split_type, 'val'),
    )    


if __name__ == "__main__" :
    main()
    sys.exit()