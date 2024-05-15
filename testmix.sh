python main.py gpu_ids=\'0\' data.bs=1 mode=test model=ssumamba \
noise=mixture test=icvl_mix test.b_size=16 \
ckpt_path=/home/fugym/data_3/papers/paper5/mamba/data/trainings/mamba/UMamba_72844_icvl_icvl_mix_MixtureNoise-mixture_0.0003/ckpts/last.ckpt \
test.save_raw=false 