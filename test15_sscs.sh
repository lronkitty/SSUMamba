python main.py gpu_ids=\'0\' data.bs=1 mode=test model=ssumamba_sscs \
noise.params.sigma_max=15 test=icvl15 test.b_size=4 \
ckpt_path=ssumamba_sscs_15.ckpt \
test.save_raw=false