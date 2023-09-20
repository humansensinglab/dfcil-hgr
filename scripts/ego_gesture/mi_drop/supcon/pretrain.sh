#!/bin/bash
. ../../../common_dirs.sh

cd $code_dir/drivers/mi_drop/supcon

drop_pretrain_single() {

    seed=$1
    gpu_ids=$2
    n_add_classes=$3
    n_known_classes=$4

    split_type=agnostic

    cfg_file=${code_dir}/configs/params/mi_drop/ego_gesture/supcon/pretrain/initial.yaml
    log_dir=$code_dir/experiments/ego_gesture/mi_drop/seed_${seed}/task_0

    rm -rf ${log_dir}

    CUDA_VISIBLE_DEVICES=${gpu_ids} \
    python main_supcon_pretrain.py \
    --dataset 'ego_gesture' \
    --split_type ${split_type} \
    --cfg_file ${cfg_file} \
    --root_dir ${ego_gesture_root_dir} \
    --log_dir ${log_dir} \
    --n_add_classes ${n_add_classes} \
    --n_known_classes ${n_known_classes} \
    --drop_seed ${seed} \
    --save_epoch_freq 10 ;

}

seed=$1
gpu_ids=$2
n_known_classes=$6
n_add_classes=$7

drop_pretrain_single \
    ${seed} \
    ${gpu_ids} \
    ${n_add_classes} \
    ${n_known_classes}