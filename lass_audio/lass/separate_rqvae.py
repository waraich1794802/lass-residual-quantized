# Utility
import argparse
from pathlib import Path
from typing import Union
from lass.utils import ROOT_DIRECTORY
# EnCodec
from encodec.encodec.model import EncodecModel
# Pytorch and dataset
import torch
from torch.utils.data import DataLoader
from lass.datasets_rqvae import MixtureDataset
# Diba interface
from lass.prior_rqvae import EncodecPrior, SparseLikelihood
# Beam Search Separator
from lass.separators_rqvae import BeamSearchSeparator

def separate(
    source_dir_1: Union[Path, str],
    source_dir_2: Union[Path, str],
    output_dir: Union[Path, str],
    prior_path_1: Union[Path, str],
    prior_path_2: Union[Path, str],
    sum_frequencies_path: Union[Path, str],
):
    '''Separate a mixture of multiple sources using lass with RQVAE

    Args:
        source_dir_1: Directory that contains the first source audio files
        source_dir_2: Directory that contains the second source audio files
        prior_path_1: Path to the first autoregressive prior
        prior_path_2: Path to the second autoregressive prior
        output_dir: Directory that will contain the separated audio files
        sum_frequencies_path: Path to the distribution of sums
    '''

    #1 Set device for separation
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
        torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    print(f"Device is: {device}");

    #2 Instantiate rqvae model from EnCodec
    rqvae = EncodecModel.encodec_model_24khz()
    rqvae.set_target_bandwidth(6.0)
    rqvae = rqvae.to(device)
    print("EnCodec model instantiated")

    #3 Instantiate prior models
    priors = [
        rqvae.get_lm_model(),
        rqvae.get_lm_model(),
    ]

    priors[0].load_state_dict(torch.load(prior_path_1))
    priors[1].load_state_dict(torch.load(prior_path_2))
    
    priors = {
        Path(prior_path_1).stem: priors[0].to(device),
        Path(prior_path_2).stem: priors[1].to(device),
    }

    #4 Instantiate separator
    separator = BeamSearchSeparator(
        encode_fn = lambda x: torch.cat([frame for frame, _ in rqvae.encode(x)], dim=-1).view(-1).tolist(),
        decode_fn = lambda x: rqvae.decode([(x, None)]),
        priors={k:EncodecPrior(p) for k,p in priors.items()},
        likelihood = SparseLikelihood(sum_frequencies_path, device, 3.0),
        num_beams = 10,
    )
    
    #5 Instantiate dataset
    mixture_dataset = MixtureDataset(source_dir_1, source_dir_2, device)
    dataset_loader = DataLoader(
        mixture_dataset,
        batch_size=1,
        #num_workers=6,
    )

    #6 Separate (and save)
    separator.separate(
        dataset_loader = dataset_loader,
        output_dir = output_dir,
    )

if __name__ == "__main__":
    #1 Define script arguments
    parser = argparse.ArgumentParser(
        description="Separate a mixture of multiple sources using lass with RQVAE"
    )
    parser.add_argument(
        "--source-dir-1",
        type=str,
        help="Directory containing samples of the first source",
        default=str(ROOT_DIRECTORY / "data/bass"),
    )
    parser.add_argument(
        "--source-dir-2",
        type=str,
        help="Directory containing samples of the second source",
        default=str(ROOT_DIRECTORY / "data/drums"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory where the separated audio will be stored",
        default=str(ROOT_DIRECTORY / "separated-audio-rqvae"),
    )
    parser.add_argument(
        "--prior-path-1",
        type=str,
        help="Path to the first autoregressive prior",
        default=str(ROOT_DIRECTORY / "checkpoints/encodec_prior_bass.pth"),
    )
    parser.add_argument(
        "--prior-path-2",
        type=str,
        help="Path to the second autoregressive prior",
        default=str(ROOT_DIRECTORY / "checkpoints/encodec_prior_drums.pth"),
    )
    parser.add_argument(
        "--sum-frequencies-path",
        type=str,
        help="Path to the distrubution of sums",
        default=str(ROOT_DIRECTORY / "checkpoints/sum_dist_rqvae.npz"),
    )

    #2 Separate with given parameters
    args = vars(parser.parse_args())
    separate(**args)