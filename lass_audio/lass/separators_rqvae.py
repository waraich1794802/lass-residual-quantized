# Utility
from typing import Callable
from pathlib import Path
# Enable Abstraction
import abc
# Pytorch and datasets
import torch
import torchaudio
from torch.utils.data import DataLoader
# Discrete Bayesian Signal Separation
import diba
from diba.diba import Likelihood, SeparationPrior
# Progress bar
from tqdm import tqdm

class Separator(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()
        
    @abc.abstractmethod
    def separate():
        ...

class BeamSearchSeparator(Separator):
    def __init__(
        self,
        encode_fn: Callable,
        decode_fn: Callable,
        likelihood: Likelihood,
        num_beams: int,
    ):
        super().__init__()
        # TODO: change to correct function
        self.encode_fn = encode_fn #lambda x: vqvae.encode(x.unsqueeze(-1), vqvae_level, vqvae_level + 1).view(-1).tolist()
        self.decode_fn = decode_fn #lambda x: decode_latent_codes(vqvae, x.squeeze(0), level=vqvae_level)

        self.likelihood = likelihood
        self.num_beams = num_beams


    @torch.no_grad()
    def separate(
        self,
        dataset_loader: DataLoader,
        output_dir: str,
    ):
        #1 Convert output directory to path and validate
        output_dir = Path(output_dir)
        if not output_dir.exists() and not len(list(output_dir.glob("*"))) == 0:
            raise ValueError(f"Path {output_dir} already exists!")

        #2 Make output directory
        output_dir.mkdir(exist_ok=True)

        #3 Loop over mixtures
        for batch_idx, batch in enumerate(tqdm(dataset_loader)):
            #3.a Convert signals to codes
            mixture_codes = self.encode_fn(batch['mixture'])

            #3.b Separate mixture (what is the shape of x?)
            #TODO: where to get encodec priors?
            diba_separated = diba.fast_beamsearch_separation(
                priors = {'first' : SeparationPrior()},
                likelihood = self.likelihood,
                mixture = mixture_codes,
                num_beams = self.num_beams,
            )

            #3.c Decode the results
            separated_sources = self.decode_fn(diba_separated)

            #3.d Save the separated sources
            #TODO: check sample rate
            torchaudio.save(str(output_dir / f"sep{batch_idx}.wav"), separated_sources.cpu(), sample_rate=22000)
