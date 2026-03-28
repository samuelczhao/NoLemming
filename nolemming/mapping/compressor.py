"""PCA compression from fsaverage5 vertex space to low-dimensional latent."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from nolemming.core.types import CompressedResponse, NeuralResponse


class VoxelCompressor:
    """Reduces 20,484-vertex activations to n_dims via PCA."""

    def __init__(self, n_dims: int = 64) -> None:
        self.n_dims = n_dims
        self.pca: PCA | None = None

    def fit(self, responses: list[NeuralResponse]) -> None:
        """Fit PCA on mean activations from multiple responses."""
        if not responses:
            msg = "Need at least one response to fit"
            raise ValueError(msg)
        data = np.stack([r.mean_activation() for r in responses])
        self._fit_data(data)

    def fit_single(self, response: NeuralResponse) -> None:
        """Fit PCA on a single response's timestep-level data."""
        self._fit_data(response.activations)

    def compress(self, response: NeuralResponse) -> CompressedResponse:
        """Compress mean activation to latent space."""
        self._check_fitted()
        mean_act = response.mean_activation().reshape(1, -1)
        latent = self.pca.transform(mean_act)[0].astype(np.float32)  # type: ignore[union-attr]
        return CompressedResponse(latent=latent)

    def compress_timesteps(
        self, response: NeuralResponse,
    ) -> NDArray[np.float32]:
        """Compress each timestep independently. Returns (n_timesteps, n_dims)."""
        self._check_fitted()
        return self.pca.transform(response.activations).astype(np.float32)  # type: ignore[union-attr]

    def _fit_data(self, data: NDArray[np.float32]) -> None:
        """Fit PCA on a 2D array of activation vectors."""
        effective_dims = min(self.n_dims, data.shape[0], data.shape[1])
        self.pca = PCA(n_components=effective_dims)
        self.pca.fit(data)

    def _check_fitted(self) -> None:
        """Raise if PCA has not been fitted yet."""
        if self.pca is None:
            msg = "Compressor not fitted. Call fit() or fit_single() first."
            raise RuntimeError(msg)
