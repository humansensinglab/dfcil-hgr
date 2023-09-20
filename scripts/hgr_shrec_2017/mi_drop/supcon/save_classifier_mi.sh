#!/bin/bash
. ../../../common_dirs.sh

cd $code_dir/drivers/mi_drop/supcon

save_classifier() {

    split_type=agnostic

    seed=$1
    gpu_ids=$2
    clf_split=$3
    eig_var_exp=$4
    cur_task_id=$5
    prev_task_id=$6
    n_add_classes=$7
    n_known_classes=$8
    inverted_data_subdir=$9

    cfg_file=${code_dir}/configs/params/mi_drop/hgr_shrec_2017/supcon/pretrain/initial_mi.yaml
    log_dir=${code_dir}/experiments/hgr_shrec_2017/mi_drop/seed_${seed}/task_${cur_task_id}
    inverted_data_dir=${code_dir}/experiments/hgr_shrec_2017/mi_drop/seed_${seed}/task_${prev_task_id}/${inverted_data_subdir}

    CUDA_VISIBLE_DEVICES=${gpu_ids} \
    python save_classifier_mi.py \
    --dataset 'hgr_shrec_2017' \
    --subset ${clf_split} \
    --split_type ${split_type} \
    --cfg_file ${cfg_file} \
    --root_dir ${hgr_shrec_2017_root_dir} \
    --log_dir ${log_dir} \
    --inverted_data_dir ${inverted_data_dir} \
    --n_add_classes ${n_add_classes} \
    --n_known_classes ${n_known_classes} \
    --drop_seed ${seed} \
    --eig_var_exp ${eig_var_exp}

}

eig_var_exp=0.95
inverted_data_subdir=inverted_samples

seed=$1
gpu_ids=$2
clf_split=$3
inv_type=$4
cur_task_id=$5
prev_task_id=$6
n_known_classes=$7
n_add_classes=$8

inverted_data_subdir=${inverted_data_subdir}_${inv_type}

save_classifier \
    ${seed} \
    ${gpu_ids} \
    ${clf_split} \
    ${eig_var_exp} \
    ${cur_task_id} ${prev_task_id} \
    ${n_add_classes} ${n_known_classes} \
    ${inverted_data_subdir}