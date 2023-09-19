# dataset dir
hgr_shrec_2017_root_dir=/ogr_cmu/data/SHREC_2017
# code dir
src_dir=/ogr_cmu/src

cd ${src_dir}/utils/preprocess
python hgr_shrec_2017_split_train_to_train_val.py --root_dir ${hgr_shrec_2017_root_dir}