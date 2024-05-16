# SSUMamba
Code of "SSUMamba: Spatial-Spectral Selective State Space Model for Hyperspectral Image Denoising"

This code is strongly based on T3SC (https://github.com/inria-thoth/T3SC)

The data can be download at https://njusteducn-my.sharepoint.com/:f:/g/personal/ironkitty_njust_edu_cn/EopcsxbL_TFJiQYEHkr9Wd8BllWJFvplYco2arbqgOhwMA?e=eqed5n .

After you download the datasets, please modify the dir path in `mamba/config/test/*.yaml`

Please install the requriment in ./ssumamba.yaml.

- Install PyTorch 2.1.1+cu118

  ```shell
  pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
  ```

- Install `causal_conv1d` and `mamba`

  ```shell
  pip install -e requirements/causal-conv1d
  pip install -e requirements/mamba
  ```