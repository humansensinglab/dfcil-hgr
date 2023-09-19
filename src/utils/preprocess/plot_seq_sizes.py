import sys
import os
import os.path as osp

import argparse

from tqdm import tqdm
from typing import Sequence

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


def get_seq_sizes(
    root_dir: str, 
    split_filepath: str,
) -> Sequence[int] :

    n_samples = get_nlines_in_file(split_filepath)
    fhand = get_file_handle(split_filepath, 'r')

    sizes_per_class = {}
    for line in tqdm( fhand, total=n_samples ) :
      
        line = line.strip()
        # sub_path, label, size_seq, id_subject = line
        _, label, size_seq, _ = line.split(',')
        label = int(label)
        size_seq = int(size_seq)
        if label not in sizes_per_class :
            sizes_per_class[label] = []
        
        sizes_per_class[label].append(size_seq)

    fhand.close()

    return [sizes_per_class[x] for x in range(len(sizes_per_class))]


def plot_seq_sizes(    
    seq_sizes_l: Sequence[Sequence[int]],
    tag_sizes_l: Sequence[str],
    label_names_l: Sequence[str],
) -> None :

    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')    
    import matplotlib.pyplot as plt

    import seaborn as sns

    label_names_l = [x.upper() for x in label_names_l]

    
    color_l = [(0.7, 0.7, 0.9), (0.9, 0.5, 0.5)]

    fig, ax = plt.subplots()
    width = 0.4
    ind = np.arange(len(seq_sizes_l[0]))
    rects1 = ax.violinplot(seq_sizes_l[0], ind - width/2, widths=width, 
                showmeans=True, showextrema=True, showmedians=True)
    rects2 = ax.violinplot(seq_sizes_l[1], ind + width/2, widths=width, 
                showmeans=True, showextrema=True, showmedians=True)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Sequence Length', fontsize=20)
    ax.set_title('Distribution of Sequence Length Per Class', fontsize=22)
    ax.set_xticks(ind, label_names_l, rotation=45, fontsize=12)
    ax.legend(['train', 'test'], fontsize=20)
    leg = ax.get_legend()
    for lh, color in zip(leg.legendHandles, color_l) :
        lh.set_color(color)

    fig.tight_layout()
    plt.show()

    seq_sizes_0 = np.array(sum(seq_sizes_l[0], []), dtype=np.float32).flatten()
    seq_sizes_1 = np.array(sum(seq_sizes_l[1], []), dtype=np.float32).flatten()

    ax = sns.histplot([seq_sizes_0, seq_sizes_1], stat='percent', kde=True)
    ax.set_ylabel('Percentage of Samples (%)', fontsize=20)
    ax.set_xlabel('Sequence Length', fontsize=20)
    ax.set_title('Distribution of Sequence Length Per Class', fontsize=22)
    ax.legend(['train', 'test'], fontsize=20)
    plt.show()


def main() :
    global args
    cfg = Config_Data(args.root_dir)    

    split_type = 'specific' # get detailed class info

    n_classes = cfg.get_n_classes(split_type)
    lname_l = [cfg.label_to_name[split_type][x] for x in range(0, n_classes)]

    class_freq_l = []
    class_freq_l.append(get_seq_sizes(
                args.root_dir, 
                cfg.get_split_filepath(split_type, 'train'),
    ) )

    class_freq_l.append(get_seq_sizes(
                args.root_dir, 
                cfg.get_split_filepath(split_type, 'test'),
    ) )

    plot_seq_sizes(
        class_freq_l,
        ['train', 'test'],
        lname_l,
    )    



if __name__ == "__main__" :
    main()
    sys.exit()