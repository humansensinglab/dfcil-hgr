#!/bin/bash
. ../../../common_dirs.sh

cd $code_dir/drivers/mi_drop/supcon

model_inversion_single() {

    seed=$1
    gpu_ids=$2

    inv_sample_subdir=$3
    inv_type=$4
    max_samples_per_class=$5
    n_add_classes=$6
    n_known_classes=$7
    cfg_file=$8
    log_dir=$9

    split_type=agnostic

    CUDA_VISIBLE_DEVICES=${gpu_ids} \
    python model_inversion.py \
    --dataset 'hgr_shrec_2017' \
    --split_type ${split_type} \
    --cfg_file ${cfg_file} \
    --root_dir ${hgr_shrec_2017_root_dir} \
    --log_dir ${log_dir} \
    --inv_sample_subdir ${inv_sample_subdir} \
    --n_add_classes ${n_add_classes} \
    --n_known_classes ${n_known_classes} \
    --inv_type ${inv_type} \
    --drop_seed ${seed} \
    --max_samples_per_class ${max_samples_per_class}
    # --use_reduced

}

seed=$1
gpu_ids=$2
clf_split=$3
inv_type=$4
max_samples_per_class=$5
n_known_classes=$6
n_add_classes=$7

cfg_file=${code_dir}/configs/params/mi_drop/hgr_shrec_2017/supcon/pretrain/initial_mi.yaml
log_dir_prefix=${code_dir}/experiments/hgr_shrec_2017/mi_drop/seed_${seed}/task_

inv_sample_subdir=inverted_samples 
task_id=0

log_dir=${log_dir_prefix}${task_id}
inv_sample_subdir=${inv_sample_subdir}_${inv_type}

model_inversion_single \
    ${seed} \
    ${gpu_ids} \
    ${inv_sample_subdir} ${inv_type} \
    ${max_samples_per_class} \
    ${n_add_classes} ${n_known_classes} \
    ${cfg_file} ${log_dir}   