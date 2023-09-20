#!/bin/bash
. ../../../common_dirs.sh

cd $code_dir/drivers/mi_drop/supcon

gen_mi_split_single() {

    seed=$1
    gpu_ids=$2
    in_dir_1=$3
    in_dir_2=$4
    out_dir=$5
    split_file_subdir=split_files

    CUDA_VISIBLE_DEVICES=${gpu_ids} \
    python gen_mi_split.py \
    --in_dir_l ${in_dir_1} ${in_dir_2} \
    --out_dir ${out_dir} \
    --split_file_subdir ${split_file_subdir}
}

seed=$1
gpu_ids=$2
clf_split=$3
inv_type=$4
cur_task_id=$5
prev_task_id=$6
n_known_classes=$7
n_add_classes=$8


log_dir_prefix=${code_dir}/experiments/hgr_shrec_2017/mi_drop/seed_${seed}/task_
inv_subdir=inverted_samples
inv_subdir_new=inverted_samples_new

inv_subdir=${inv_subdir}_${inv_type}
inv_subdir_new=${inv_subdir_new}_${inv_type}

gen_mi_split_single \
    ${seed} \
    ${gpu_ids} \
    ${log_dir_prefix}${prev_task_id}/${inv_subdir} \
    ${log_dir_prefix}${cur_task_id}/${inv_subdir_new} \
    ${log_dir_prefix}${cur_task_id}/${inv_subdir}

