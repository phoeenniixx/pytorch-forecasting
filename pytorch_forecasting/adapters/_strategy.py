import pandas as pd
import torch

from pytorch_forecasting.adapters.utils import (
    ArrayLike,
    _series_from,
    _to_numpy,
    _to_tensor,
    _was_2d_singleton,
)


class _ScalerStrategy:
    """Default behavior for scalers."""

    is_label_encoder = False
    fit_per_sequence = False

    def prepare_input(self, data: ArrayLike) -> ArrayLike:
        t = _to_tensor(data)
        return t.squeeze(-1) if (t.ndim == 2 and t.shape[1] == 1) else t

    def fit(self, scaler, prepared: ArrayLike, X: pd.DataFrame = None) -> None:
        scaler.fit(prepared)

    def transform(
        self, scaler, prepared: ArrayLike, data: ArrayLike, X: pd.DataFrame = None
    ) -> torch.Tensor:
        result = _to_tensor(scaler.transform(prepared))
        return result.unsqueeze(-1) if _was_2d_singleton(data) else result


class _EncoderNormalizerStrategy(_ScalerStrategy):
    """EncoderNormalizer must be re-fit per encoder window."""

    fit_per_sequence = True


class _SklearnStrategy(_ScalerStrategy):
    """sklearn scalers expect/return 2D numpy arrays."""

    def prepare_input(self, data: ArrayLike) -> ArrayLike:
        return _to_numpy(data).reshape(-1, 1)

    def fit(self, scaler, prepared: ArrayLike, X: pd.DataFrame = None) -> None:
        scaler.fit(prepared)

    def transform(
        self, scaler, prepared: ArrayLike, data: ArrayLike, X: pd.DataFrame = None
    ) -> torch.Tensor:
        original_shape = _to_numpy(data).shape
        result = scaler.transform(prepared).reshape(original_shape)
        return torch.tensor(result, dtype=torch.float32)


class _LabelEncoderStrategy(_ScalerStrategy):
    is_label_encoder = True

    def prepare_input(self, data: ArrayLike) -> ArrayLike:
        return _series_from(data)

    def transform(
        self, scaler, prepared: ArrayLike, data: ArrayLike, X: pd.DataFrame = None
    ) -> torch.Tensor:
        result = _to_tensor(scaler.transform(prepared))
        return result.unsqueeze(-1) if _was_2d_singleton(data) else result


class _GroupNormalizerStrategy(_ScalerStrategy):
    requires_group_columns = True

    def prepare_input(self, data: ArrayLike) -> ArrayLike:
        return _series_from(data)

    def fit(self, scaler, prepared: ArrayLike, X: pd.DataFrame = None) -> None:
        assert X is not None, (
            "GroupNormalizer requires X (DataFrame with group columns) "
            "to be passed to fit()."
        )
        scaler.fit(prepared, X)

    def transform(
        self, scaler, prepared: ArrayLike, data: ArrayLike, X: pd.DataFrame = None
    ) -> torch.Tensor:
        assert X is not None, (
            "GroupNormalizer requires X (DataFrame with group columns) "
            "to be passed to transform()."
        )
        input_was_2d = isinstance(data, torch.Tensor) and data.ndim == 2
        result = _to_tensor(scaler.transform(prepared, X))
        return result.unsqueeze(-1) if input_was_2d else result
