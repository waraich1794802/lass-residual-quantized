# Source Separation in Residual Quantized Latent Domains
This is the repo for the *Latent Autoregressive Source Separator*, as described in its [paper](https://arxiv.org/abs/2301.08562), applied to the residual setting. It is an based upon the [original repo](https://github.com/gladia-research-group/latent-autoregressive-source-separation/tree/main).

## Installation
Before running the original code it was necessary to install an MPI implementation. Refer to the old repo for detailed instruction.

This version is instead designed to run on single device, be it a cpu or a cuda compatible gpu.

Installing the necessary dependencies can be done through conda, by executing the following commands:
```bash
cd lass_audio
conda env create -f environment.yml
conda activate lass_audio
``` 
Once the conda environment is installed it is possible to start the separation procedure or the training. The command line will inform you of any missing arguments.

## Download Pre-trained models and data
You can download the necessary checkpoints from [here](https://drive.google.com/drive/folders/1Hc3XtJBsXVu8zH-8Aj2I_z0grZKWeQUk?usp=sharing).
Place the downloaded file inside of the directory `lass_audio/checkpoints` and you can optionally rename them to "encodec_prior_bass.pth", "encodec_prior_drums.pth" and "sum_dist_rqvae.npz" (see next section for details).

The Slakh2100 test data can be found [here](https://drive.google.com/file/d/1Gf5SHVb8_o5NMbJAWaoULianX8RxzL4L/view?usp=share_link).
Place the downloaded file inside of the directory `lass_audio/data` and extract it with the command `tar -xf bass-drums-slakh.tar`

## Separate images
If you renamed the checkpoints you can simply run
```bash
PYTHONPATH=. python lass\separate_rqvae.py
``` 
To separate with the best settings you can run the script
```bash
PYTHONPATH=. python lass\separate_rqvae.py --prior-path-1 checkpoints\encodec_prior_3500_bass.pth --prior-path-2 checkpoints\encodec_prior_34000_drums.pth --sum-frequencies-path checkpoints\sum_dist_rqvae_29000.npz
``` 
On Windows two commands are necessary
```bash
set PYTHONPATH=.
python lass\separate_rqvae.py --prior-path-1 checkpoints\encodec_prior_3500_bass.pth --prior-path-2 checkpoints\encodec_prior_34000_drums.pth --sum-frequencies-path checkpoints\sum_dist_rqvae_29000.npz
``` 

