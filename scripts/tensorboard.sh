port=9998
log_dir=/ogr_cmu/output/

mkdir -p ${log_dir}

CUDA_VISIBLE_DEVICES=0 tensorboard --logdir=${log_dir}  --port=${port} --bind_all