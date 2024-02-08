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
from lass.diba_interfaces import SparseLikelihood
# Beam Search Separator
from lass.separators_rqvae import BeamSearchSeparator


def separate(
    source_dir_1: Union[Path, str],
    source_dir_2: Union[Path, str],
    output_dir: Union[Path, str],
    sum_frequencies_path: Union[Path, str],
):
    '''Separate a mixture of multiple sources using lass with RQVAE

    Args:
        source_dir_1: Directory that contains the first source audio files
        source_dir_2: Directory that contains the second source audio files
        output_dir: Directory that will contain the separated audio files
        sum_frequencies_path: Path to the distribution of sums
    '''

    #1 Set device for separation
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(0)
    print(f"Device is: {device}");

    #2 Instantiate rqvae model from EnCodec
    rqvae = EncodecModel.encodec_model_24khz()
    rqvae = rqvae.to(device)
    print("EnCodec model instantiated")

    #3 Instantiate separator
    separator = BeamSearchSeparator(
        encode_fn = lambda x: 2*x,
        decode_fn = lambda x: 0.5*x,
        likelihood = SparseLikelihood(sum_frequencies_path, device, 3.0),
        num_beams = 10,
    )
    
    #4 Instantiate dataset
    mixture_dataset = MixtureDataset(source_dir_1, source_dir_2)
    dataset_loader = DataLoader(
        mixture_dataset,
        batch_size=1,
        num_workers=6,
    )

    #5 Separate (and save)
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
        default=str(ROOT_DIRECTORY / "separated-audio"),
    )
    parser.add_argument(
        "--sum-frequencies-path",
        type=str,
        help="Path to the distrubution of sums",
        default=str(ROOT_DIRECTORY / "checkpoints/sum_frequencies.npz"),
    )

    #2 Separate with given parameters
    args = vars(parser.parse_args())
    separate(**args)