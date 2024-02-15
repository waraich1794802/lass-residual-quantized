# Utility
import argparse
import re
import sparse
from pathlib import Path
from typing import Union, Tuple
from lass.utils import ROOT_DIRECTORY
# EnCodec
from encodec.encodec.model import EncodecModel
# Prior (Unused in this version)
from lass.prior_rqvae import EncodecPriorModel
# Pytorch, Numpy and dataset
import torch
import numpy
import torch.nn.functional as F
from torch.utils.data import DataLoader
from lass.datasets_rqvae import TrainDataset
# Progress bar
from tqdm import tqdm

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
    '''Train an autoregressive prior of the latent codes of a RQ-VAE

    Args:
        epochs: Number of epochs used for estimation
        batch_size: Batch size used during estimation
        save_iters: A checkpoint is stored each #save_iters iterations
        train_dir: Directory that contains the training audio files
        output_dir: Directory that will contain the estimated distribution file
        checkpoint_path: Path of the checkpoint used for resuming estimation
    '''
    
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

    #3 Instantiate prior
    prior = rqvae.get_lm_model()
    prior.train()
    prior.to(device)
    print("Prior model instantiated")
    
    # Used in a previous version:
    #prior = EncodecPriorModel(rqvae)
    #print(f"Codebook size: {prior.bins}");
    #print(f"Number of quantizers: {prior.n_q}");

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
    if checkpoint_path is not None:
        prior.load_state_dict(torch.load(checkpoint_path))
        prior.train()
        prior.to(device)

    #6 Initialize loop variables and optimizer
    iterations = 0
    loss = 0
    optimizer = torch.optim.Adam(prior.parameters(), lr=0.0002)

    #7 Run training loop
    print("Starting training")
    for epoch in range(epochs):
        print(f"Epoch {epoch}/{epochs}");
        print("loss: {0}".format(loss)) #print loss of the last training batch, not fully indicative of actual trend
        for batch_idx, batch in enumerate(tqdm(dataset_loader)):
            
            optimizer.zero_grad()
            
            #7.a blend channels randomly to mono (augmentation used in the original lass)
            mix = torch.rand((batch.shape[0],1), device=batch.device)
            x = (mix*batch[:,:,0] + (1-mix)*batch[:,:,1])
            x = x[:,None,:].to(device)

            #7.b get the latent codes from the RQ-VAE
            with torch.no_grad():
                frames = rqvae.encode(x)
                codes = torch.cat([f for f, _ in frames], dim=-1)

            #7.c apply the autoregressive prior on the codes
            # add 1 because index 0 is reserved for start token
            probabilities, _, _ = prior(codes + 1)

            #7.d gradient descent
            loss = F.cross_entropy(probabilities, codes)
            loss.backward()
            optimizer.step()

            #7.e free unused memory
            del probabilities, codes, frames, x, mix
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            #7.f save model every #save_iters iterations
            iterations += 1
            if iterations % save_iters == 0:
                # save prior model
                output_dir=Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = output_dir / f"encodec_prior_{iterations}.pth"
                torch.save(prior.state_dict(), checkpoint_path)

if __name__ == "__main__":
    #1 Define script arguments
    parser = argparse.ArgumentParser(
        description="Train an autoregressive prior on the latent codes of a RQ-VAE"
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
        default =str(ROOT_DIRECTORY / "data/bass"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory where the prior model will be stored",
        default=str(ROOT_DIRECTORY / "logs/rqvae_priors"),
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