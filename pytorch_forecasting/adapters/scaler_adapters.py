from typing import Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
import torch

from pytorch_forecasting.data.encoders import (
    EncoderNormalizer,
    MultiNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
)

_SKLEARN_SCALERS = (RobustScaler, StandardScaler)

ArrayLike = torch.Tensor | np.ndarray | pd.Series


def _to_numpy(data: ArrayLike) -> np.ndarray:
    """Convert any array-like to numpy."""
    if isinstance(data, torch.Tensor):
        return data.detach().numpy()
    elif isinstance(data, pd.Series):
        return data.to_numpy()
    return np.asarray(data)


def _to_tensor(data: ArrayLike, dtype=torch.float32) -> torch.Tensor:
    """Convert any array-like to a float32 tensor."""
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    elif isinstance(data, pd.Series):
        return torch.tensor(data.to_numpy(), dtype=dtype)
    return torch.tensor(np.asarray(data), dtype=dtype)


class ScalerAdapter:
    """
    Unified array-in / tensor-out interface for single and multi-target scalers.

    Accepts torch.Tensor, np.ndarray, or pd.Series as input. Output is always
    a torch.Tensor. Conversions are performed only when the underlying scaler
    requires a specific type:

    - sklearn scalers     : need numpy, reshaped to (-1, 1)
    - GroupNormalizer     : needs pd.Series (handled by MultiNormalizer internally)
    - TorchNormalizer /
      EncoderNormalizer /
      NaNLabelEncoder     : work natively with tensors
    - MultiNormalizer     : delegates per-column; each sub-normalizer gets what it needs

    Parameters
    ----------
    scaler : sklearn scaler, TorchNormalizer, EncoderNormalizer, NaNLabelEncoder,
             or MultiNormalizer
    """

    def __init__(self, scaler):
        self._scaler = scaler
        self.is_multi = isinstance(scaler, MultiNormalizer)
        self._is_sklearn = isinstance(scaler, _SKLEARN_SCALERS)

        if self.is_multi:
            self._sub_adapters = [ScalerAdapter(norm) for norm in scaler.normalizers]

    @property
    def fit_per_sequence(self) -> bool:
        """True if any normalizer must be (re-)fit at __getitem__ time."""
        if self.is_multi:
            return any(a.fit_per_sequence for a in self._sub_adapters)
        return isinstance(self._scaler, EncoderNormalizer)

    def _prepare_input(self, data: ArrayLike) -> ArrayLike:
        """Coerce data to the type the underlying scaler expects.

        - sklearn         : numpy, shape (-1, 1)
        - torch-native    : tensor, squeezed to 1-D if (N, 1)
        - MultiNormalizer : tensor or numpy 2-D (N, n_targets) — MultiNormalizer
                            handles its own per-column dispatch internally
        """
        if self._is_sklearn:
            return _to_numpy(data).reshape(-1, 1)

        if self.is_multi:
            if isinstance(data, torch.Tensor):
                return data if data.ndim == 2 else data.unsqueeze(-1)
            arr = _to_numpy(data)
            return arr if arr.ndim == 2 else arr[:, None]

        t = _to_tensor(data)
        return t.squeeze(-1) if (t.ndim == 2 and t.shape[1] == 1) else t

    def fit(self, data: ArrayLike) -> "ScalerAdapter":
        """Fit the scaler.

        Parameters
        ----------
        data : tensor, ndarray, or Series
            Shape ``(n_samples,)`` for single-target or
            ``(n_samples, n_targets)`` for multi-target.
        """
        if self._scaler is None:
            return self
        self._scaler.fit(self._prepare_input(data))
        return self

    def transform(self, data: ArrayLike) -> torch.Tensor:
        """Transform data, always returning a torch.Tensor.

        Parameters
        ----------
        data : tensor, ndarray, or Series
            Shape ``(n_samples,)`` for single-target or
            ``(n_samples, n_targets)`` for multi-target.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        if self._scaler is None:
            return _to_tensor(data)

        prepared = self._prepare_input(data)

        if self._is_sklearn:
            original_shape = _to_numpy(data).shape
            result = self._scaler.transform(prepared).reshape(original_shape)
            return torch.tensor(result, dtype=torch.float32)

        if self.is_multi:
            results = self._scaler.transform(prepared.T)
            return torch.stack([_to_tensor(r) for r in results], dim=-1)

        squeezed = (
            isinstance(data, torch.Tensor) and data.ndim == 2 and data.shape[1] == 1
        )
        result = self._scaler.transform(prepared)
        result = _to_tensor(result)
        return result.unsqueeze(-1) if squeezed else result

    def fit_transform(self, data: ArrayLike) -> torch.Tensor:
        return self.fit(data).transform(data)

    def fit_transform_sequence(self, data: ArrayLike) -> torch.Tensor:
        """Fit-and-transform only per-sequence sub-normalizers; transform the rest.

        Used at ``__getitem__`` time for encoder windows. Non-per-sequence
        normalizers use their already-fitted global state.

        For single-target adapters this collapses to fit_transform
        (EncoderNormalizer) or transform (everything else).

        Parameters
        ----------
        data : tensor, ndarray, or Series
            Shape ``(enc_length,)`` or ``(enc_length, n_targets)``.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        if not self.is_multi:
            return (
                self.fit_transform(data)
                if self.fit_per_sequence
                else self.transform(data)
            )

        t = _to_tensor(data)
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        columns = []
        for idx, sub in enumerate(self._sub_adapters):
            col = t[:, idx]
            col = sub.fit_transform(col) if sub.fit_per_sequence else sub.transform(col)
            columns.append(col.unsqueeze(-1))
        return torch.cat(columns, dim=-1)
