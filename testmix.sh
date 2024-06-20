python main.py gpu_ids=\'0\' data.bs=1 mode=test model=ssumamba \
noise=mixture test=icvl_mix test.b_size=16 \
ckpt_path=ssumamba_mix.ckpt \
test.save_raw=false 