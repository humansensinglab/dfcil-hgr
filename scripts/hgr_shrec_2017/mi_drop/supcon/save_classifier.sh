#!/bin/bash
. ../../../common_dirs.sh

cd $code_dir/drivers/mi_drop/supcon

save_classifier() {

    split_type=agnostic

    seed=$1
    gpu_ids=$2
    clf_split=$3
    eig_var_exp=$4
    n_add_classes=$5
    n_known_classes=$6

    cfg_file=${code_dir}/configs/params/mi_drop/hgr_shrec_2017/supcon/pretrain/initial.yaml
    log_dir=$code_dir/experiments/hgr_shrec_2017/mi_drop/seed_${seed}/task_0

    CUDA_VISIBLE_DEVICES=${gpu_ids} \
    python save_classifier.py \
    --dataset 'hgr_shrec_2017' \
    --subset ${clf_split} \
    --split_type ${split_type} \
    --cfg_file ${cfg_file} \
    --root_dir ${hgr_shrec_2017_root_dir} \
    --log_dir ${log_dir} \
    --n_add_classes ${n_add_classes} \
    --n_known_classes ${n_known_classes} \
    --drop_seed ${seed} \
    --eig_var_exp ${eig_var_exp}

}

seed=$1
gpu_ids=$2
clf_split=$3
inv_type=$4
n_known_classes=$6
n_add_classes=$7

eig_var_exp=0.95


save_classifier \
    ${seed} \
    ${gpu_ids} \
    ${clf_split} \
    ${eig_var_exp} \
    ${n_add_classes} \
    ${n_known_classes}