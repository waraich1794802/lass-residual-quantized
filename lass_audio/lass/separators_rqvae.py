# Utility
from typing import Callable, Mapping
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
        priors: Mapping[str, SeparationPrior],
        likelihood: Likelihood,
        num_beams: int,
    ):
        super().__init__()
        self.encode_fn = encode_fn #TODO, put the function here once is finalized
        self.decode_fn = decode_fn #lambda x: rqvae.decode(x)

        self.source_types = list(priors)
        self.priors = list(priors.values())
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
            #print("mixture.shape is: {0}".format(batch['mixture'].shape))
            mixture_codes = self.encode_fn(batch['mixture'])
            #print("len(mixture_codes) is: {0}".format(len(mixture_codes)))
            #print("mixture.type is: {0}".format(mixture_codes.type()))

            #3.b Separate mixture
            #TODO: what is the shape of diba_separated?
            #TODO: doesn't make sense that mixture_codes are int?
            diba_separated_1, diba_separated_2 = diba.fast_beamsearch_separation(
                priors = self.priors,
                likelihood = self.likelihood,
                mixture = mixture_codes,
                num_beams = self.num_beams,
            )

            #3.c Decode the results
            separated_source_1 = self.decode_fn(diba_separated_1)
            separated_source_2 = self.decode_fn(diba_separated_2)

            #3.d Save the separated sources
            #TODO: check sample rate
            torchaudio.save(str(output_dir / f"mix{batch_idx}.wav"), batch['mixture'].cpu(), sample_rate=22000)
            torchaudio.save(str(output_dir / f"sep{batch_idx}_1.wav"), separated_source_1.cpu(), sample_rate=22000)
            torchaudio.save(str(output_dir / f"sep{batch_idx}_2.wav"), separated_source_2.cpu(), sample_rate=22000)
