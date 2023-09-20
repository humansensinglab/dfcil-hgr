#!/bin/bash

seed=$1
gpu_ids=$2
out_dir=$3
clf_split=$4
inv_type=$5
max_samples_per_class=$6
n_known_classes=0
n_add_classes=$7

args=(
    ${seed} 
    ${gpu_ids} 
    ${clf_split} 
    ${inv_type} 
    ${max_samples_per_class}
    ${n_known_classes} 
    ${n_add_classes}    
)

./save_proto.sh ${args[@]}
./model_inversion.sh ${args[@]} 