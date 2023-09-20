#!/bin/bash
. ../../../common_dirs.sh

cd $code_dir/drivers/mi_drop/supcon

pretrain_mi_single() {

    seed=$1
    gpu_ids=$2
    cur_task_id=$3
    prev_task_id=$4
    n_add_classes=$5
    n_known_classes=$6
    inverted_data_subdir=$7

    split_type=agnostic

    cfg_file=${code_dir}/configs/params/mi_drop/hgr_shrec_2017/supcon/pretrain/initial_mi.yaml
    log_dir=${code_dir}/experiments/hgr_shrec_2017/mi_drop/seed_${seed}/task_${cur_task_id}
    pretrain_dir=${code_dir}/experiments/hgr_shrec_2017/mi_drop/seed_${seed}/task_${prev_task_id}

    rm -rf ${log_dir}

    CUDA_VISIBLE_DEVICES=${gpu_ids} \
    python main_supcon_pretrain_mi.py \
    --dataset 'hgr_shrec_2017' \
    --split_type ${split_type} \
    --cfg_file ${cfg_file} \
    --root_dir ${hgr_shrec_2017_root_dir} \
    --log_dir ${log_dir} \
    --pretrain_dir ${pretrain_dir} \
    --inverted_data_dir ${pretrain_dir}/${inverted_data_subdir} \
    --n_add_classes ${n_add_classes} \
    --n_known_classes ${n_known_classes} \
    --drop_seed ${seed} \
    --save_epoch_freq 10 ;

}

inverted_data_subdir=inverted_samples

seed=$1
gpu_ids=$2
clf_split=$3
inv_type=$4
cur_task_id=$5
prev_task_id=$6
n_known_classes=$7
n_add_classes=$8

# echo ${clf_split}
# echo ${inv_type}
# echo ${cur_task_id}
# echo ${prev_task_id}
# echo ${n_known_classes}
# echo ${n_add_classes}

inverted_data_subdir=${inverted_data_subdir}_${inv_type}

pretrain_mi_single \
    ${seed} \
    ${gpu_ids} \
    ${cur_task_id} ${prev_task_id} \
    ${n_add_classes} ${n_known_classes} \
    ${inverted_data_subdir}