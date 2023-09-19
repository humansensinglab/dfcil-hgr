# dataset dir
hgr_shrec_2017_root_dir=/ogr_cmu/data/SHREC_2017
# code dir
src_dir=/ogr_cmu/src

cd ${src_dir}/utils/preprocess
python gen_hgr_shrec_2017_dataset.py --root_dir ${hgr_shrec_2017_root_dir}
