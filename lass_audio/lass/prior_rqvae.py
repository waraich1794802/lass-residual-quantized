# Utility
import math
import functools
import sparse
from typing import Any, Optional, Tuple
# EnCodec and Encodec prior
from encodec.encodec.model import EncodecModel, LMModel
# Prior for RQ-Transformer
from jukebox.prior.autoregressive import ConditionalAutoregressive2D
# Pytorch and numpy
import torch
import numpy as np
# Diba interfaces
from diba.diba import Likelihood
from diba.interfaces import SeparationPrior

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
        x = torch.sum(x, 0).unsqueeze(0)
            
        # Compute the loss for the spatial prior
        loss, _, _ = self.spatial_prior(x)

        # Compute the loss

        return loss

class EncodecPrior(SeparationPrior):
    def __init__(self, transformer: LMModel):
        self._prior = transformer

    @functools.lru_cache(1)
    def get_device(self) -> torch.device:
        return list(self._prior.transformer.parameters())[0].device

    #TODO: change to correct out_features
    def get_tokens_count(self) -> int:
        print("i'm outputting: {0}".format(self._prior.linears[0].out_features * 8))
        return self._prior.linears[0].out_features * 8

    def get_sos(self) -> Any:
        return 0 #return start token as defined in LMModel

    def get_logits(
            self, token_ids: torch.LongTensor, cache: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, Optional[Any]]:

        # assert token lenght is > 0
        assert len(token_ids) > 0
        
        # if sos token then we create proper start token
        if len(token_ids[0]) == 1:
            token_ids = torch.zeros(1, 8, 1).long().to(self.get_device())
        # else add 1 to differentiate index 0 from start token
        else:
            token_ids = token_ids.view() + 1
            
        # get dimensions
        print("token_ids shape is: {0}".format(token_ids.shape))
        n_samples, n_q, seq_length = token_ids.shape
        sample_t = seq_length - 1
        #print(f"token is: {token_ids}");

        x = token_ids[:,:, -1:]
        print(f"x shape is: {x.shape}");

        # TODO: change order to match the likelihood matrix
        #apply transformer
        x, _, _ = self._prior(x)
        x = x[:,:,:,-1].view(n_samples, -1)
        #print(f"output shape is: {x.shape}");

        return x.to(torch.float32), None

    def reorder_cache(self, cache: Any, beam_idx: torch.LongTensor) -> Any:
        #TODO: understand what this is for
        #self._prior.transformer.substitute_cache(beam_idx)
        return None


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
        sparse_nll = self._freqs[:, :, token_idx].tocoo()
        nll_coords = torch.tensor(sparse_nll.coords, device=self.get_device(), dtype=torch.long)
        nll_data = torch.tensor(sparse_nll.data, device=self.get_device(), dtype=torch.float)
        return nll_coords, nll_data

    def _normalize_matrix(self, sum_dist_path: str):
        sum_dist = sparse.load_npz(str(sum_dist_path))
        print("Likelihood matrix's shape is: {0}".format(sum_dist.shape))
        #sum_dist = sum_dist[:1024,:1024,:1024]
        #TODO: CHANGE ABOVE TO BE CORRECT
        integrals = sum_dist.sum(axis=-1, keepdims=True)
        I, J, _ = integrals.coords
        integrals = sparse.COO(
            integrals.coords, integrals.data, shape=integrals.shape, fill_value=1
        )
        log_data = np.log(sum_dist / integrals) * self._lambda_coeff
        return sparse.GCXS.from_coo(log_data, compressed_axes=[2])

