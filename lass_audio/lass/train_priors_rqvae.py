# Utility
import argparse
import re
import sparse
from pathlib import Path
from typing import Union, Tuple
from lass.utils import ROOT_DIRECTORY
# EnCodec
from encodec.encodec.model import EncodecModel
# Prior
from jukebox.prior.prior import SimplePrior
from jukebox.prior.autoregressive import ConditionalAutoregressive2D
# Pytorch and dataset
import torch
from torch.utils.data import DataLoader
from lass.datasets_rqvae import TrainDataset
# Progress bar
from tqdm import tqdm

def make_prior(rqvae: EncodecModel) -> ConditionalAutoregressive2D:
    return ConditionalAutoregressive2D(
        input_shape = [8,32,1024],
        bins = 1024
    )

    return SimplePrior(
        z_shapes = 'missing',
        l_bins = rqvae.quantizer.bins,
        encoder = rqvae.encode,
        decoder = rqvae.decode,
        level = 'missing',
        downs_t = 'missing',
        strides_t = 'missing',
        labels = False,
        prior_kwargs = 'missing',
        x_cond_kwargs = 'missing',
        y_cond_kwargs = 'missing',
        prime_kwargs = 'missing',
        copy_input = False,
        labels_v3 = False,
        merged_decoder = False,
        single_enc_dec = False,
    )

# pickle-able collate function
def collate_pickle(batch):
    return torch.stack([torch.from_numpy(b) for b in batch], 0)

def train_prior(
    epochs: int,
    batch_size: int,
    save_iters: int,
    train_dir: Union[Path, str],
    output_dir: Union[Path, str],
    checkpoint_path: Union[Path, str, None] = None,
):
    '''Train an autoregressive prior on top of the latent codes of a RQ-VAE

    Args:
        epochs: Number of epochs used for estimation
        batch_size: Batch size used during estimation
        save_iters: A checkpoint is stored each #save_iters iterations
        train_dir: Directory that contains the training audio files
        output_dir: Directory that will contain the estimated distribution file
        checkpoint_path: Path of the checkpoint used for resuming estimation
    '''
    
    #0 Define helper functions
    #0.a to load the prior model
    #TODO: load model instead of matrix
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
        assert isistance(sum_dist, sparse.COO)
        return sum_dist, iterations
    
    #1 Set device for training
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(0)
    print(f"Device is: {device}");
    
    #2 Instantiate rqvae model from EnCodec
    rqvae = EncodecModel.encodec_model_24khz()
    rqvae = rqvae.to(device)
    print("EnCodec model instantiated")
    print(f"Codebook size: {rqvae.quantizer.bins}");
    print(f"Number of quantizers: {rqvae.quantizer.n_q}");

    #3 Instantiate prior
    prior = make_prior(rqvae)
    prior = prior.to(device)
    print("Prior model instantiated")

    #4 Instantiate data-loader
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
    
    #5 Load checkpoint if available
    # TODO
    prefix_i, prefix_j, prefix_k, prefix_data = [], [], [], []
    iterations = 0

    #6 Initialize loop variables
    buffer_add_1, buffer_add_2, buffer_sum = [], [], []

    #7 Run training loop
    print("Starting training")
    with torch.no_grad():
        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs}");
            for batch_idx, batch in enumerate(tqdm(dataset_loader)):
                #7.a blend channels randomly to mono
                mix = torch.rand((batch.shape[0],1), device=batch.device)
                x = (mix*batch[:,:,0] + (1-mix)*batch[:,:,1])
                x = x[:,None,:].to(device)

                #7.b
                print(f"Type is: {x.type()}");
                frames = rqvae.encode(x)
                codes = torch.cat([f for f, _ in frames], dim=-1)
                prior_codes = prior(codes)
                decoded = rqvae.decode(prior_codes)

                #7.c compute loss and backpropagate through prior

                #7.d save model every #save_iters iterations
                iterations += 1
                if iterations % save_iters == 0:
                    # save prior model
                    output_dir=Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = output_dir / f"sum_dist_rqvae_{iterations}.npz"

                    # load prior model?
                    sum_dist, iterations = load_checkpoint(checkpoint_path)
                    prefix_i, prefix_j, prefix_k = sum_dist.coords.tolist()
                    prefix_data = sum_dist.data.tolist()
                    del sum_dist

                    # reset loop variables
                    buffer_add_1, buffer_add_2, buffer_sum = [], [], []

if __name__ == "__main__":
    #1 Define script arguments
    parser = argparse.ArgumentParser(
        description="Train an autoregressive prior on top of the latent codes of a RQ-VAE"
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
        help="Directory where the prior model will be stored",
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
    train_prior(**args)