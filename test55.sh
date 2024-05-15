python main.py gpu_ids=\'0\' data.bs=1 mode=test model=ssumamba \
noise.params.sigma_max=55 test=icvl55 test.b_size=1 \
ckpt_path=/home/fugym/data_3/papers/paper5/mamba/data/trainings/mamba/UMamba_72843_icvl_icvl_55_UniformNoise-mi0-ma55_0.0003/ckpts/last.ckpt \
test.save_raw=false