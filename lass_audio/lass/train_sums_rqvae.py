# Utility
import argparse
import re
import sparse
import math
from pathlib import Path
from typing import Union, Tuple
from lass.utils import ROOT_DIRECTORY
# EnCodec
from encodec.encodec.model import EncodecModel
# Pytorch, Numpy and dataset
import torch
import numpy as np
from torch.utils.data import DataLoader
from lass.datasets_rqvae import TrainDataset
# Progress bar
from tqdm import tqdm

# pickle-able collate function
def collate_pickle(batch):
    return torch.stack([torch.from_numpy(b) for b in batch], 0)

def estimate_distribution(
    epochs: int,
    batch_size: int,
    alpha: Tuple[float],
    save_iters: int,
    train_dir: Union[Path, str],
    output_dir: Union[Path, str],
    checkpoint_path: Union[Path, str, None] = None,
):
    '''Estimate frequencies of sum-result pairs in pretrained RQ-VAE

    Args:
        epochs: Number of epochs used for estimation
        batch_size: Batch size used during estimation
        alpha: Convex coefficients for the mixture
        save_iters: A checkpoint is stored each #save_iters iterations
        train_dir: Directory that contains the training audio files
        output_dir: Directory that will contain the estimated distribution file
        checkpoint_path: Path of the checkpoint used for resuming estimation
    '''
    
    #0 Define helper functions
    #0.a to compute latent codes in batches
    def compute_latent(batch, rqvae):
        frames = rqvae.encode(batch)
        codes = torch.cat([f for f, _ in frames], dim=-1)
        
        # codes are in the form [B, K, T]
        # B is the batch size
        # K is the number of codebooks
        # T is the codebook size
        codes = codes.cpu().numpy()
        concatenated_codes = np.reshape(codes, newshape=-1, order="F").tolist()
        return concatenated_codes

    #0.b to load the distribution of sums
    def load_checkpoint(checkpoint_path):
        # check format and get #iterations
        checkpoint_filename = Path(checkpoint_path).name
        match = re.fullmatch(r"sum_dist_rqvae_(?P<iterations>[0-9]+)\.npz", checkpoint_filename)
        if match is None:
            raise RuntimeError(
            f"The filename {checkpoint_filename} is not in the correct format!"
            "It must be of format 'sum_dist_[NUM_ITER].npz'!"
            )
        iterations = int(match.group("iterations"))

        # load sparse matrix
        sum_dis = sparse.load_npz(checkpoint_path)
        assert isinstance(sum_dist, sparse.COO)
        return sum_dist, iterations
    
    #1 Set device for training
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
    
    #2.a Compute number of latent bins * depth
    depth = int(1000 * rqvae.bandwidth // (math.ceil(rqvae.sample_rate / rqvae.encoder.hop_length) * 10))
    bins_x_depth = rqvae.quantizer.bins * depth
    print(f"Codebook size: {rqvae.quantizer.bins}");
    print(f"Number of quantizers: {depth}");

    #3 Instantiate data-loader
    train_dataset = TrainDataset(train_dir)
    dataset_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=6,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_pickle,
    )
    
    #4 Load checkpoint if available
    if checkpoint_path is not None:
        sum_dist, iterations = load_checkpoint(checkpoint_path)
        prefix_i, prefix_j, prefix_k = sum_dist.coords.tolist()
        prefix_data = sum_dist.data.tolist()
        del sum_dist
    else:
        prefix_i, prefix_j, prefix_k, prefix_data = [], [], [], []
        iterations = 0

    #5 Initialize loop variables
    buffer_add_1, buffer_add_2, buffer_sum = [], [], []
    
    #6 Run estimation (training) loop
    print("Starting training")
    with torch.no_grad():
        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs}");
            for batch_idx, batch in enumerate(tqdm(dataset_loader)):
                #6.a blend channels randomly to mono
                mix = torch.rand((batch.shape[0],1), device=batch.device)
                x = (mix*batch[:,:,0] + (1-mix)*batch[:,:,1])
                x = x[:,None,:]

                #6.b split batch and sum the two halves to make mixtures
                x_1 = x[: batch_size // 2].to(device)
                x_2 = x[batch_size // 2 :].to(device)
                x_sum = alpha[0] * x_1 + alpha[1] * x_2

                #6.c compute latent vectors using rq-vae
                buffer_add_1.extend(compute_latent(x_1, rqvae))
                buffer_add_2.extend(compute_latent(x_2, rqvae))
                buffer_sum.extend(compute_latent(x_sum, rqvae))

                #6.d save output every #save_iters iterations
                iterations += 1
                if iterations % save_iters == 0:
                    sum_dist = sparse.COO(
                        coords=[
                            prefix_i + buffer_add_1 + buffer_add_2,
                            prefix_j + buffer_add_2 + buffer_add_1,
                            prefix_k + buffer_sum + buffer_sum,
                        ],
                        data=prefix_data + [1] * (len(buffer_add_1) * 2),
                        shape=[bins_x_depth, bins_x_depth, bins_x_depth],
                    )

                    # store results as a sparse matrix
                    output_dir=Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = output_dir / f"sum_dist_rqvae_{iterations}.npz"
                    sparse.save_npz(str(checkpoint_path), sum_dist)

                    # load sparse matrix
                    sum_dist, iterations = load_checkpoint(checkpoint_path)
                    prefix_i, prefix_j, prefix_k = sum_dist.coords.tolist()
                    prefix_data = sum_dist.data.tolist()
                    del sum_dist

                    # reset loop variables
                    buffer_add_1, buffer_add_2, buffer_sum = [], [], []

if __name__ == "__main__":
    #1 Define script arguments
    parser = argparse.ArgumentParser(
        description="Compute an approximate distribution of sums of latent codes in the residual setting"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs",
        default=100
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
        default=8
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs=2,
        help="Convex coefficients for the mixture",
        default=(0.5, 0.5),
        metavar=("ALPHA_1", "ALPHA_2"),
    )
    parser.add_argument(
        "--save-iters",
        type=int,
        help="Interval of steps before saving partial results",
        default=500,
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        help="Directory containing the training audio files",
        default =str(ROOT_DIRECTORY / "data/train"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory where the output estimation will be stored",
        default=str(ROOT_DIRECTORY / "logs/rqvae_sum_distribution"),
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Checkpoint path",
        default=None
    )

    #2 Estimate distribution with parsed arguments
    args = vars(parser.parse_args())
    estimate_distribution(**args)