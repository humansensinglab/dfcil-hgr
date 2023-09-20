#!/bin/bash

clf_split=val

inv_type=proto-svm

max_samples_per_class=300

seed=$1
gpu_ids=$2

out_dir_prefix=tmp/seed_

n_known_classes=8 # base classes
n_add_classes=1 # each time
n_tasks=6 # excluding base task

echo "Running with seed = ${seed} on GPU = ${gpu_ids}"

out_dir=${out_dir_prefix}${seed}
mkdir -p ${out_dir}

args=(
    ${seed} 
    ${gpu_ids} 
    ${out_dir} 
    ${clf_split} 
    ${inv_type} 
    ${max_samples_per_class} 
    ${n_known_classes} 
    ${n_add_classes}
    ${n_tasks}
)

# ./base_build.sh ${args[@]}
# ./base_runner.sh ${args[@]}
./mi_runner.sh ${args[@]}

