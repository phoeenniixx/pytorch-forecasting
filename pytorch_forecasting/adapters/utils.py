import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
import torch

_SKLEARN_SCALERS = (RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler)

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


def is_sklearn_scaler(scaler):
    return isinstance(scaler, _SKLEARN_SCALERS)
