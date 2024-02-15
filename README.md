# Source Code for Source Separation in Residual Quantized Latent Domains  - 
Deep Learning Project [2022/23]

This is the repo for the deep learning project involving adapting lass to the residual setting.
The code of interest is in the folder * [`./lass_audio`](https://github.com/waraich1794802/lass-residual-quantized/tree/main/lass_audio) 
The added files are as follows:
```lua
lass_audio
|-- lass
|   |-- datasets_rqvae.py
|   |-- prior_rqvae.py
|   |-- separate_rqvae.py
|   |-- separators_rqvae.py
|   |-- train_priors_rqvae.py
|   |-- train_sums_rqvae.py
```
Diba interfaces have also been modifies accordingly.
Quick links to the main scripts:
[`separate_rqvae.py`](https://github.com/waraich1794802/lass-residual-quantized/blob/main/lass_audio/lass/separate_rqvae.py) 
[`train_priors_rqvae.py`](https://github.com/waraich1794802/lass-residual-quantized/blob/main/lass_audio/lass/train_priors_rqvae.py) 
[`train_sums_rqvae.py`](https://github.com/waraich1794802/lass-residual-quantized/blob/main/lass_audio/lass/train_sums_rqvae.py) 

See the `README.md` in the subdirectory for more info on how to install the environment and run experiments.

  **NOTE: to install the necessary environment it is recommended the user to 
  use [conda](https://docs.conda.io/en/latest/miniconda.html) as environment manager for python.**

