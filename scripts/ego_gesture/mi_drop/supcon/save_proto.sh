#!/bin/bash
. ../../../common_dirs.sh

cd $code_dir/drivers/mi_drop/supcon

save_proto_single() {

    split_type=agnostic

    seed=$1
    gpu_ids=$2

    clf_split=$3

    eig_var_exp=$4
    n_add_classes=$5
    n_known_classes=$6

    cfg_file=$7
    log_dir=$8    

    CUDA_VISIBLE_DEVICES=${gpu_ids} \
    python save_proto.py \
    --dataset 'ego_gesture' \
    --subset ${clf_split} \
    --split_type ${split_type} \
    --cfg_file ${cfg_file} \
    --root_dir ${ego_gesture_root_dir} \
    --log_dir ${log_dir} \
    --eig_var_exp ${eig_var_exp} \
    --n_add_classes ${n_add_classes} \
    --n_known_classes ${n_known_classes} \
    --drop_seed ${seed}        

}

seed=$1
gpu_ids=$2
clf_split=$3
inv_type=$4
max_samples_per_class=$5
n_known_classes=$6
n_add_classes=$7

cfg_file=${code_dir}/configs/params/mi_drop/ego_gesture/supcon/pretrain/initial.yaml
log_dir_prefix=${code_dir}/experiments/ego_gesture/mi_drop/seed_${seed}/task_
eig_var_exp=0.95

task_id=0
save_proto_single \
    ${seed} \
    ${gpu_ids} \
    ${clf_split} \
    ${eig_var_exp} \
    ${n_add_classes} ${n_known_classes} \
    ${cfg_file} ${log_dir_prefix}${task_id}