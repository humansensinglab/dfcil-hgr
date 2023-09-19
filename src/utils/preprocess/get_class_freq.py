import sys
import os
import os.path as osp

import importlib
import argparse

from tqdm import tqdm
from typing import Sequence
from pprint import pprint

if os.path.dirname(sys.argv[0]) != '':
    os.chdir(os.path.dirname(sys.argv[0]))
sys.path.append(osp.join('..', '..'))

from utils.stdio import *

parser = argparse.ArgumentParser(description='Generate train/val/test splits for EgoGesture.')
parser.add_argument('--root_dir', default='/ogr_cmu/data/SHREC_2017', type=str, 
                    help='root directory containing the dataset.')
parser.add_argument('--dataset', default='hgr_shrec_2017', type=str, 
                    help='name of the dataset.')                    

args = parser.parse_args()
print_argparser_args(args)


def get_samples_per_class(
    root_dir: str, 
    split_filepath: str,
) -> Sequence[int] :

    n_samples = get_nlines_in_file(split_filepath)
    fhand = get_file_handle(split_filepath, 'r')

    samples_per_class = {}
    for line in tqdm( fhand, total=n_samples ) :
      
        line = line.strip()
        # sub_path, label, size_seq, id_subject = line
        _, label, _, _ = line.split(',')
        label = int(label)
        samples_per_class[label] = samples_per_class.get(label, 0) + 1

    fhand.close()

    assert n_samples == sum(samples_per_class.values())

    pprint(samples_per_class)
    return [samples_per_class[x] for x in range(len(samples_per_class))]


def plot_class_freq(    
    freq_l: Sequence[Sequence[int]],
    tag_freq_l: Sequence[str],
    label_names_l: Sequence[str],
    save_fname: str,
) -> None :

    import numpy as np
    import matplotlib
    # matplotlib.use('TkAgg')    
    import matplotlib.pyplot as plt

    label_names_l = [x.upper() for x in label_names_l]

    
    color_l = [(0.5, 0.5, 0.9), (0.9, 0.5, 0.5), (0.5, 0.9, 0.5)]

    fig, ax = plt.subplots()
    ind = np.arange(len(freq_l[0]))    
    for i in range(len(freq_l)) :
        ax.plot(ind, freq_l[i], color=color_l[i])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('#Samples', fontsize=14)
    ax.set_title('Distribution of Samples Per Class', fontsize=16)
    ax.legend(tag_freq_l)

    fig.tight_layout()
    fig.savefig(save_fname)


def main() :
    global args
    cfg = getattr(importlib.import_module(
                    '.' + args.dataset, 
                    package='configs.datasets'),
                'Config_Data')(args.root_dir) 

    split_type = 'specific' # get detailed class info

    n_classes = cfg.get_n_classes(split_type)
    if split_type in cfg.label_to_name :
        label_to_name = cfg.label_to_name[split_type]
    else :
        label_to_name = cfg.label_to_name

    lname_l = [label_to_name[x] for x in range(0, n_classes)]


    class_freq_l = []
    mode_l = ['train', 'test']
    for mode in mode_l :
        class_freq_l.append(get_samples_per_class(
                    args.root_dir, 
                    cfg.get_split_filepath(split_type, mode),
        ) )

    save_fname = args.dataset + '_class_distribution.png'
    plot_class_freq(
        class_freq_l,
        mode_l,
        lname_l,
        save_fname,
    )    


if __name__ == "__main__" :
    main()
    sys.exit()