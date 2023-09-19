src_dir=$1
datasets=$2
methods=$3

cd ${src_dir}/utils

###################### Generate LaTex tables ########################
for dataset_name in ${datasets[*]}; do
    python latex.py $dataset_name "$methods"
done