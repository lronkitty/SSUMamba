python main.py gpu_ids=\'0\' data.bs=1 mode=test model=ssumamba \
noise.params.sigma_max=95 test=icvl95 test.b_size=1 \
ckpt_path=ssumamba_95.ckps \
test.save_raw=false