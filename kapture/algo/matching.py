# Copyright 2020-present NAVER Corp. Under BSD 3-clause license

from abc import ABC, abstractmethod
import torch
import numpy as np
from kapture.utils.logging import getLogger


class MatchPairGenerator(ABC):
    @abstractmethod
    def match_descriptors(self, descriptors_1, descriptors_2):
        raise NotImplementedError()


class MatchPairNnTorch(MatchPairGenerator):
    def __init__(self, use_cuda=True):
        super().__init__()
        self._device = torch.device("cuda:0"
                                    if use_cuda and torch.cuda.is_available()
                                    else "cpu")

    def match_descriptors(self, descriptors_1, descriptors_2):
        if descriptors_1.shape[0] == 0 or descriptors_2.shape[0] == 0:
            return np.zeros((0, 3))

        # send data to GPU
        descriptors1_torch = torch.from_numpy(descriptors_1).to(self._device)
        descriptors2_torch = torch.from_numpy(descriptors_2).to(self._device)
        # make sure its double (because CUDA tensors only supports floating-point)
        descriptors1_torch = descriptors1_torch.float()
        descriptors2_torch = descriptors2_torch.float()
        # sanity check
        if not descriptors1_torch.device == self._device:
            getLogger().debug('descriptor on device {} (requested {})'.format(descriptors1_torch.device, self._device))
        if not descriptors2_torch.device == self._device:
            getLogger().debug('descriptor on device {} (requested {})'.format(descriptors2_torch.device, self._device))

        simmilarity_matrix = descriptors1_torch @ descriptors2_torch.t()
        scores = torch.max(simmilarity_matrix, dim=1)[0]
        nearest_neighbor_idx_1vs2 = torch.max(simmilarity_matrix, dim=1)[1]
        nearest_neighbor_idx_2vs1 = torch.max(simmilarity_matrix, dim=0)[1]
        ids1 = torch.arange(0, simmilarity_matrix.shape[0], device=descriptors1_torch.device)
        # cross check
        mask = ids1 == nearest_neighbor_idx_2vs1[nearest_neighbor_idx_1vs2]
        matches_torch = torch.stack(
            [ids1[mask].type(torch.float), nearest_neighbor_idx_1vs2[mask].type(torch.float), scores[mask]]).t()
        # retrieve data back from GPU
        matches = matches_torch.data.cpu().numpy()
        matches = matches.astype(np.float)
        return matches
