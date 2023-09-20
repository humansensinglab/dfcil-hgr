#!/bin/bash

seed=$1
gpu_ids=$2
out_dir=$3
clf_split=$4
inv_type=$5
max_samples_per_class=$6
n_known_classes=$7
n_add_classes=$8
n_tasks=$9

out_file_prefix=${out_dir}/sp_${clf_split}_itype_${inv_type}_mspc_${max_samples_per_class}_acc_

start_task_id=1

for (( cur_task_id=${start_task_id}; cur_task_id<=${n_tasks}; ++cur_task_id )) ; do
    prev_task_id=$((cur_task_id-1))

    args=(
        ${seed}
        ${gpu_ids} 
        ${clf_split} 
        ${inv_type} 
        ${cur_task_id} 
        ${prev_task_id} 
        ${n_known_classes} 
        ${n_add_classes}
    )

    out_file=${out_file_prefix}${cur_task_id}.log

    ./pretrain_mi.sh ${args[@]}
    ./save_classifier_mi.sh ${args[@]}
    ./test_classifier_mi.sh ${args[@]} > ${out_file}

    if [ ! ${cur_task_id} -eq ${n_tasks} ]; then
        # for next task
        ./save_proto_mi.sh ${args[@]}
        ./model_inversion_mi.sh ${args[@]} ${max_samples_per_class} 
        ./gen_mi_split.sh ${args[@]}
    fi

    n_known_classes=$((n_known_classes + n_add_classes))

done