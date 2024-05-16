python main.py gpu_ids=\'0\' data.bs=1 mode=test model=ssumamba \
noise.params.sigma_max=55 test=icvl55 test.b_size=1 \
ckpt_path=ssumamba_55.ckps \
test.save_raw=false