#!/bin/bash

# Run experiment
src_dir=/ogr_cmu/src
scripts_dir=/ogr_cmu/scripts
cd ${scripts_dir}

# General config
split_type="agnostic"
CUDA_VISIBLE_DEVICES=0
gpu=0

#datasets=("hgr_shrec_2017"  "ego_gesture")
datasets=("hgr_shrec_2017")
#baselines=("Base" "Fine_tuning" "Feature_extraction" "LwF" "LwF.MC" "DeepInversion" "ABD" "Rdfcil")
baselines=("Base")
trial_ids=(0 1 2)
n_trials=${#trial_ids[@]}
n_tasks=7

#Run trials
for trial_id in ${trial_ids[*]}; do
    ./run_trial.sh $trial_id $src_dir $split_type $gpu "${datasets[*]}" "${baselines[*]}"
done

# Summarize results
./summarize_results.sh $src_dir "${datasets[*]}" "${baselines[*]}" $n_trials $n_tasks

# Generate LaTex tables
./generate_latex.sh $src_dir "${datasets[*]}" "${baselines[*]}"



