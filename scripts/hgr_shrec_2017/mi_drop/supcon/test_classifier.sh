#!/bin/bash
. ../../../common_dirs.sh

cd $code_dir/drivers/mi_drop/supcon

test_classifier_single() {

    split_type=agnostic
    seed=$1
    gpu_ids=$2
    n_add_classes=$3
    n_known_classes=$4

    cfg_file=${code_dir}/configs/params/mi_drop/ego_gesture/supcon/pretrain/initial.yaml
    log_dir=$code_dir/experiments/ego_gesture/mi_drop/seed_${seed}/task_0

    CUDA_VISIBLE_DEVICES=${gpu_ids} \
    python test_classifier.py \
    --dataset 'ego_gesture' \
    --split_type ${split_type} \
    --cfg_file ${cfg_file} \
    --root_dir ${ego_gesture_root_dir} \
    --log_dir ${log_dir} \
    --n_add_classes ${n_add_classes} \
    --n_known_classes ${n_known_classes} \
    --drop_seed ${seed}
}

seed=$1
gpu_ids=$2
clf_split=$3
inv_type=$4
n_known_classes=$6
n_add_classes=$7


test_classifier_single \
    ${seed} \
    ${gpu_ids} \
    ${n_add_classes} \
    ${n_known_classes}