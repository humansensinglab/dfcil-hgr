import sys
import os
import os.path as osp

import argparse

from tqdm import tqdm

if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))

sys.path.append(osp.join('..', '..'))

from configs.datasets.hgr_shrec_2017 import Config_Data
from utils.stdio import *

parser = argparse.ArgumentParser(description='Generate train/val/test splits for HGR SHREC 2017.')
parser.add_argument('--root_dir', default='/ogr_cmu/data/SHREC_2017' ,type=str, 
                    help='root directory containing the dataset.')

args = parser.parse_args()
print_argparser_args(args)


def get_info_from_line(line, split_type) :
    def __is_file_valid(split_type, id_finger) :
        if split_type == 'single' :
            return id_finger == 1
        elif split_type == 'multiple' :
            return id_finger > 1
        
        # don't care for agnostic/specific
        return True


    def __get_label_wrt_type(split_type, label_14, label_28) :
        if split_type == 'specific' :
            return label_28 - 1 # 0-based
        
        return label_14 - 1 # 0-based 

    line = line.strip()    
    id_gesture, id_finger, id_subject, id_trial, \
            label_14, label_28, size_seq = \
            list(map(int, line.split()) )
    
    if not __is_file_valid(split_type, id_finger) :
        return None

    sub_path = osp.join(
            f"gesture_{id_gesture}",
            f"finger_{id_finger}",
            f"subject_{id_subject}",
            f"essai_{id_trial}")

    label = __get_label_wrt_type(split_type, label_14, label_28)            

    return sub_path, label, size_seq, id_subject


def process_single_subset(
    root_dir: str, 
    raw_split_filepath: str,
    split_filepath: str,
    split_type: str,
) -> None :

    n_samples = get_nlines_in_file(raw_split_filepath)
    fin = get_file_handle(raw_split_filepath, 'r')
    fout = get_file_handle(split_filepath, 'w+')

    count = 0
    crop_len = len(root_dir) + 1


    for line in tqdm( fin, total=n_samples ) :

        line_info = get_info_from_line(line, split_type)
        if line_info is None :
            continue
        
        sub_path, label, size_seq, id_subject = line_info
        skel_path = osp.join(root_dir, sub_path, 'skeletons_world.txt')
        assert osp.isfile(skel_path), \
            f"Skeleton file not found = {skel_path}"

        fout.write(
            skel_path[crop_len:] + ',' + 
            str(label) + ',' + 
            str(size_seq) + ',' + 
            str(id_subject) + '\n' 
        )
            
        count += 1

    fin.close()
    fout.close()
    print("Number of samples = {}".format(count) )



def main() :
    global args
    cfg = Config_Data(args.root_dir)    

    for stype in cfg.split_types :
        for mode in cfg.raw_split_files :
            
            print(f"Processing '{stype}' split, '{mode}' set ...")
            process_single_subset(
                    args.root_dir, 
                    cfg.raw_split_files[mode], 
                    cfg.get_split_filepath(stype, mode),
                    stype, 
            )



if __name__ == "__main__" :
    main()
    sys.exit()