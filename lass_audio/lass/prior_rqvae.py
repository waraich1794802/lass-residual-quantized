# Utility
import math
import functools
import sparse
from typing import Any, Optional, Tuple, Sequence
# EnCodec and Encodec prior
from encodec.encodec.model import EncodecModel, LMModel
# Prior for RQ-Transformer
from jukebox.prior.autoregressive import ConditionalAutoregressive2D
# Pytorch and numpy
import torch
import numpy as np
# Diba interfaces
from .diba.diba.diba import Likelihood
from .diba.diba.interfaces import SeparationPrior

class EncodecPriorModel(torch.nn.Module):
    def __init__(self, rqvae: EncodecModel):
        super().__init__()
        self.bins = rqvae.quantizer.bins
        self.n_q = int(1000 * rqvae.bandwidth // (math.ceil(rqvae.sample_rate / rqvae.encoder.hop_length) * 10))

        self.spatial_prior = ConditionalAutoregressive2D(
            input_shape = [1024],
            bins = self.bins * self.n_q,
            width = 1024, #1024 original
            depth = 48, #48 original
        )

        self.depth_prior = ConditionalAutoregressive2D(
            input_shape = [self.n_q],
            bins = self.n_q,
            width = 8, #1024 original
            depth = 2, #48 original
        )

    def forward(self, x):
        x = x.view(self.n_q, self.bins)
        x_sp = torch.sum(x, 0).unsqueeze(0)
        x_dp = x.permute(0, 1)
            
        # Compute the tokens for the spatial prior
        loss, x_sp = self.spatial_prior(x_sp)

        # Compute the tokens for the depth prior
        loss, x_dp = self.depth_prior(x_dp)

        return loss

class EncodecPrior(SeparationPrior):
    def __init__(self, transformer: LMModel):
        self._prior = transformer

    @functools.lru_cache(1)
    def get_device(self) -> torch.device:
        return list(self._prior.transformer.parameters())[0].device

    def get_tokens_count(self) -> int:
        return self._prior.linears[0].out_features * 8

    def get_sos(self) -> Any:
        return torch.zeros(10, 8).to(self.get_device()) #return start token as defined in LMModel

    def get_logits(
            self, token_ids: torch.LongTensor, cache: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:

        #print("token_ids shape is: {0}".format(token_ids.shape))
        
        # assert token lenght is > 0
        assert len(token_ids) > 0

        # get dimensions
        #print("token_ids shape is: {0}".format(token_ids.shape))
        n_samples, n_q, seq_length = token_ids.shape
        sample_t = seq_length - 1
        if cache is not None:
            state, offset = cache
        else:
            state, offset = (None, 0)

        #print(f"token is: {token_ids}");

        # add 1 to differentiate index 0 from start token
        token_ids = token_ids + 1

        # take last element
        x = token_ids[:,:, -1:]
        #print(f"x shape is: {x.shape}");

        #apply transformer
        x, state, offset = self._prior(x, state, offset)
        #print(f"x shape after prior is: {x.shape}");
        x = x[:,:,:,-1]
        #TODO: this might be an issue
        x = x.view(n_samples, -1)
        #print(f"output shape is: {x.shape}");

        return x.to(torch.float32), (state, offset)

    def reorder_cache(self, cache: Any, beam_idx: torch.LongTensor) -> Any:
        return cache


class SparseLikelihood(Likelihood):
    def __init__(self, sum_dist_path: str, device: torch.device, lambda_coeff: float = 1.0):
        self._device = torch.device(device)
        self._lambda_coeff = lambda_coeff
        self._freqs = self._normalize_matrix(sum_dist_path)


    def get_device(self) -> torch.device:
        return self._device

    def get_tokens_count(self) -> int:
        return self._freqs.shape[0]

    @functools.lru_cache(512)
    def get_log_likelihood(self, token_idx: int) -> Tuple[torch.LongTensor, torch.Tensor]:
        #print("_freqs shape is: {0}".format(self._freqs.shape))
        sparse_nll = self._freqs[:, :, token_idx].tocoo()
        nll_coords = torch.tensor(sparse_nll.coords, device=self.get_device(), dtype=torch.long)
        nll_data = torch.tensor(sparse_nll.data, device=self.get_device(), dtype=torch.float)
        return nll_coords, nll_data

    def _normalize_matrix(self, sum_dist_path: str):
        sum_dist = sparse.load_npz(str(sum_dist_path))
        print("Likelihood matrix's shape is: {0}".format(sum_dist.shape))
        integrals = sum_dist.sum(axis=-1, keepdims=True)
        I, J, _ = integrals.coords
        integrals = sparse.COO(
            integrals.coords, integrals.data, shape=integrals.shape, fill_value=1
        )
        log_data = np.log(sum_dist / integrals) * self._lambda_coeff
        return sparse.GCXS.from_coo(log_data, compressed_axes=[2])

