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
from .diba import diba
from .diba.diba.diba import Likelihood, SeparationPrior
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
        self.encode_fn = encode_fn #lambda x: torch.cat([frame for frame, _ in rqvae.encode(x)], dim=-1).view(-1).tolist()d
        self.decode_fn = decode_fn #lambda x: rqvae.decode([(x, None)])

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
            if torch.sum(batch) == 0:
                continue
            torchaudio.save(str(output_dir / f"mix_{batch_idx}.wav"), batch[0].cpu(), sample_rate=24000)
            #3.a Convert signals to codes
            #print("mixture.shape is: {0}".format(batch['mixture'].shape))
            mixture_codes = self.encode_fn(batch)
            #print("mixture_codes with scale (?): {0}".format(mixture_codes))
            #print("len(mixture_codes) is: {0}".format(len(mixture_codes)))
            #print("mixture.type is: {0}".format(mixture_codes.type()))

            #3.b Separate mixture
            diba_separated_1, diba_separated_2 = diba.fast_beamsearch_separation(
                priors = self.priors,
                likelihood = self.likelihood,
                mixture = mixture_codes,
                num_beams = self.num_beams,
            )

            print("diba_separated_1 is: {0}".format(diba_separated_1.shape))
            print("diba_separated_2 is: {0}".format(diba_separated_2.shape))

            #3.c Decode the results
            separated_source_1 = self.decode_fn(diba_separated_1)
            separated_source_2 = self.decode_fn(diba_separated_2)
            
            #3.d This is to verify that the priors work correctly
            codes = torch.Tensor(mixture_codes).view(1, 8, -1).long().cuda()
            transformed_0 = torch.argmax(self.priors[0]._prior(codes + 1)[0], 1)
            transformed_1 = torch.argmax(self.priors[1]._prior(codes + 1)[0], 1)
            decoded_mixture_0 = self.decode_fn(transformed_0)
            decoded_mixture_1 = self.decode_fn(transformed_1)
            torchaudio.save(str(output_dir / f"dec0_{batch_idx}.wav"), decoded_mixture_0[0].cpu(), sample_rate=24000)
            torchaudio.save(str(output_dir / f"dec1_{batch_idx}.wav"), decoded_mixture_1[0].cpu(), sample_rate=24000)
            
            #3.e Save the separated sources
            torchaudio.save(str(output_dir / f"sep1_{batch_idx}.wav"), separated_source_1[0].cpu(), sample_rate=24000)
            torchaudio.save(str(output_dir / f"sep2_{batch_idx}.wav"), separated_source_2[0].cpu(), sample_rate=24000)
