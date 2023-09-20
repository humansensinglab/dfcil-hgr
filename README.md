## Data-Free Class-Incremental Hand Gesture Recognition
Code and dataset for the ICCV 2023 paper:\
**Data-Free Class-Incremental Hand Gesture Recognition**\
S. Aich, J Ruiz-Santaquiteria, Z. Lu, P. Garg, K J Joseph, A. F. Garcia, V. N. Balasubramanian, K. Kin, C. Wan, N. C. Camgoz, S. Ma, and F De la Torre\
International Conference on Computer Vision (ICCV), 2023\
[[project]]

This (main) branch contains the implementation of the method proposed in this paper. Please refer to the [baseline branch](https://github.com/humansensinglab/dfcil-hgr/tree/baseline) for the other DFCIL methods used for benchmarking.

### Training

* Make sure the directories are correct for you in the ./scripts/common_dirs.sh file.

* Go to the corresponding scripts directory for a particular dataset:
```
$ cd scripts/<DATASET>/mi_drop/supcon
$ # <DATASET> is hgr_shrec_2017 or ego_gesture
```

* Run with a particular seed:
```
$ ./all_runner_seed.sh <SEED> <GPU_ID>
$ ./all_runner_seed.sh 0 1 # seed 0 gpu 1
```

* The accuracies with incremental class indices will be logged into the ./scripts/<DATASET>/mi_drop/supcon/tmp directory.

* Good luck!

### Datasets

* [EgoGesture3D](https://drive.google.com/file/d/1pHE0Q9MtVS5BLaV2CBN1rLP_Ed7nvfac/view?usp=drive_link): The license for the EgoGesture3D skeletons are the same as this repository (license.txt). Please refer to the [EgoGesture](https://ieeexplore.ieee.org/document/8299578) paper and the website for the original video dataset and corresponding license.
* [SHREC-2017 train/val/test splits](https://drive.google.com/file/d/1o5T1b_jUG-czGp-xsGOFaVgzEJNEMnmh/view?usp=drive_link): This zip file only contains the split files comprising the list of files. Please refer to the [SHREC 2017 website](http://www-rech.telecom-lille.fr/shrec2017-hand/) to download the dataset.


[project]: http://humansensing.cs.cmu.edu/node/551


### Citation

If you find this paper/code useful, please consider citing (.bib): 


```
@InProceedings{boat-mi-dfcil,
    author    = {Aich, Shubhra and Ruiz-Santaquiteria, Jesus and Lu, Zhenyu and Garg, Prachi and Joseph, KJ and Fernandez Garcia, Alvaro and Balasubramanian, Vineeth N and Kin, Kenrick and Wan, Chengde and Camgoz, Necati Cihan and Ma, Shugao and De la Torre, Fernando},
    title     = {Data-Free Class-Incremental Hand Gesture Recognition},
    booktitle = ICCV,
    year      = {2023},
}
```