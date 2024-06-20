python main.py gpu_ids=\'0\' data.bs=1 mode=test model=ssumamba_sscs \
noise.params.sigma_max=95 test=icvl95 test.b_size=1 \
ckpt_path=ssumamba_sscs_95.ckpt \
test.save_raw=false