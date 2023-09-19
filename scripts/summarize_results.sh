#!/bin/bash

# Run experiment
src_dir=$1
datasets=$2
baselines=$3
n_trials=$4
n_tasks=$5

cd ${src_dir}/drivers

run_driver() {

    python summarize_results.py \
    --root_log_dir ${root_log_dir} \
    --n_trials ${n_trials} \
    --n_tasks ${n_tasks} \

}

for dataset_name in ${datasets[*]}; do

    for baseline_name in ${baselines[*]}; do
        ############################ Run baseline ############################
        root_log_dir=/ogr_cmu/output/$dataset_name/$baseline_name
        n_trials=$n_trials
        n_tasks=$n_tasks
        run_driver
    done
done


