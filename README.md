
# Baseline benchmark of BOAT-MI

  

### ICCV 2023

  

This is a subbranch of the Paper [Data-Free Class-Incremental Hand Gesture Recognition], which contains the Pytorch implementation of all the baseline approaches.

## Dataset preparation

The data preparation step is the same as what's documented in the repository of our proposed method.
- and replace the dataset directory `root_dir` in `run_trial.sh` with your own local dataset directory
```bash
for  dataset_name  in ${datasets[*]}; do

if [ $dataset_name  =  "hgr_shrec_2017" ]

then

dataset="hgr_shrec_2017"

root_dir="/ogr_cmu/data/SHREC_2017"

elif [ $dataset_name  =  "ego_gesture" ]

then

dataset="ego_gesture"

root_dir="/ogr_cmu/data/ego_gesture_v4"

fi
```  

## Training

  

Three seeds are randomly picked to run three experiments for each baseline approach.
- You may choose to rerun the whole experiments on your own, but we store the checkpoint of initial pre-trained model for all three seeds to make a fair comparison between different approach. and the saved checkpoint can be download from https://drive.google.com/drive/folders/1gsIPd-BGXvb2zVIWRZwDZ75ejxtxcV05?usp=sharing
	- Place the pre-trained model folder `models` under the parent directory `ogr_cmu`
	- Skip this step if you want to run the experiments entirely

- Run all experiments by one command

```

./scripts/run_experiments_all.sh

```

- Run single specific experiments by simply changing some configurations in the `run_experiments_all.sh` file. For example, run ABD approach on Shrec-2017 for one trial.

```bash

split_type="agnostic"

CUDA_VISIBLE_DEVICES=0

gpu=0

datasets=("hgr_shrec_2017")

baselines=("Oracle")

trial_ids=(0)

n_trials=${#trial_ids[@]}

n_tasks=1

```

## License

  
  

## Citation

  

If you use find this paper/code useful, please consider citing:

  

```

  

```

## Acknowledgments

The structure of our code is inspired by [ Always Be Dreaming](https://github.com/GT-RIPL/AlwaysBeDreaming-DFCIL).